import torch
from torch.utils import data, model_zoo
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp
import random
import argparse
from model.deeplab import Deeplab
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.idd_dataset import IDDDataSet

BATCH_SIZE = 8
NUM_STEPS = 30000

SOURCE = 'GTA5'  # 'GTA5' or 'SYNTHIA'
DIR_NAME = 'Baseline_L_G_18'

if SOURCE == 'GTA5':
    DATA_DIRECTORY = '/work/GTA5'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'
    NUM_CLASSES = 18
elif SOURCE == 'SYNTHIA':
    DATA_DIRECTORY = '/work/SYNTHIA'
    DATA_LIST_PATH = './dataset/synthia_list/train.txt'
    NUM_CLASSES = 13

RESTORE_FROM_RESNET = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
LEARNING_RATE = 2.5e-4

def get_arguments():
    parser = argparse.ArgumentParser(description="ACE pre-training")
    parser.add_argument("--source", type=str, default=SOURCE)
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--set", type=str, default='train')
    parser.add_argument("--save-pred-every", type=int, default=5000)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--random-seed", type=int, default=1338)
    parser.add_argument("--dir-name", type=str, default=DIR_NAME)
    parser.add_argument("--restore-from-resnet", type=str, default=RESTORE_FROM_RESNET,
                        help="Where restore model parameters from.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    return parser.parse_args()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, args):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        for i in range(1, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, base_lr, iter, max_iter, power=0.9):
    lr = lr_poly(base_lr, iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr

def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (512, 256)

    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create network
    model = Deeplab(args=args)
    saved_state_dict = model_zoo.load_url(args.restore_from_resnet)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        # Scale.layer5.conv2d_list.3.weight
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
    model.load_state_dict(new_params)

    model.train()
    model.to(device)

    # Dataloader
    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                        crop_size=input_size, ignore_label=args.ignore_label,
                        set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    elif args.source == 'SYNTHIA':
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.batch_size,
                           crop_size=input_size, ignore_label=args.ignore_label,
                           set=args.set, num_classes=args.num_classes),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError('Unavailable source domain')
    trainloader_iter = enumerate(trainloader)

    # implement model.optim_parameters(args) to handle different models' lr setting
    optimizer = optim.SGD(model.optim_parameters(args), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005)
    optimizer.zero_grad()

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # start training
    for i_iter in range(args.num_steps):

        loss_seg_value = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter, args)

        # train G
        _, batch = trainloader_iter.__next__()

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.long().to(device)

        pred = model(images, input_size)

        loss_seg = seg_loss(pred, labels)
        loss = loss_seg
        loss_seg_value += loss_seg.item()

        loss.backward()

        optimizer.step()

        print('exp = {}'.format(osp.join(args.snapshot_dir, args.dir_name)))
        print(
            'iter = {0:8d}/{1:8d}, loss_seg = {2:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value))

        # Snapshots directory
        if not os.path.exists(osp.join(args.snapshot_dir, args.dir_name)):
            os.makedirs(osp.join(args.snapshot_dir, args.dir_name))

        # Save model
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(),
                       osp.join(args.snapshot_dir, args.dir_name, str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.dir_name, str(i_iter) + '.pth'))

if __name__ == '__main__':
    main()
