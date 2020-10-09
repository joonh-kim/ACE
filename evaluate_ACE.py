import argparse
import numpy as np
import random

import torch
from torch.utils import data
from model.deeplab import Deeplab
from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
from dataset.crosscity_dataset import CrossCityDataSet

FILENAME = './snapshots/ACE_GR/'

GTA5 = True
SYNTHIA = False
Rio = False
Rome = False
Taipei = False
Tokyo = False

PER_CLASS = True

SAVE_PRED_EVERY = 5000
NUM_STEPS_STOP = 50000

BATCH_SIZE = 6

DATA_DIRECTORY_GTA5 = '/work/GTA5'
DATA_LIST_PATH_GTA5 = './dataset/gta5_list/val.txt'

DATA_DIRECTORY_SYNTHIA = '/work/SYNTHIA'
DATA_LIST_PATH_SYNTHIA = './dataset/synthia_list/val.txt'

DATA_DIRECTORY_TARGET = '/work/NTHU_Datasets'

NUM_CLASSES = 13

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def get_arguments():
    parser = argparse.ArgumentParser(description="ACE evaluation")
    parser.add_argument("--gta5", action='store_true', default=GTA5)
    parser.add_argument("--synthia", action='store_true', default=SYNTHIA)
    parser.add_argument("--mIoUs-per-class", action='store_true', default=PER_CLASS)
    parser.add_argument("--data-dir-gta5", type=str, default=DATA_DIRECTORY_GTA5)
    parser.add_argument("--data-list-gta5", type=str, default=DATA_LIST_PATH_GTA5)
    parser.add_argument("--data-dir-synthia", type=str, default=DATA_DIRECTORY_SYNTHIA)
    parser.add_argument("--data-list-synthia", type=str, default=DATA_LIST_PATH_SYNTHIA)
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--set", type=str, default='val')
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--random-seed", type=int, default=1338)

    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")

    return parser.parse_args()


def main():
    args = get_arguments()

    seed = args.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    input_size = (512, 256)

    name_classes = np.asarray(["road",
                               "sidewalk",
                               "building",
                               "light",
                               "sign",
                               "vegetation",
                               "sky",
                               "person",
                               "rider",
                               "car",
                               "bus",
                               "motorcycle",
                               "bicycle"])

    # Create the model and start the evaluation process
    model = Deeplab(args=args)
    for files in range(int(args.num_steps_stop / args.save_pred_every)):
        print('Step: ', (files + 1) * args.save_pred_every)
        saved_state_dict = torch.load(FILENAME + str((files + 1) * args.save_pred_every) + '.pth')
        model.load_state_dict(saved_state_dict)

        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        model.eval()
        if args.gta5:
            gta5_loader = torch.utils.data.DataLoader(
                GTA5DataSet(args.data_dir_gta5, args.data_list_gta5,
                            crop_size=input_size, ignore_label=args.ignore_label,
                            set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(gta5_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (GTA5): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if args.synthia:
            synthia_loader = torch.utils.data.DataLoader(
                SYNTHIADataSet(args.data_dir_synthia, args.data_list_synthia,
                               crop_size=input_size, ignore_label=args.ignore_label,
                               set=args.set, num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(synthia_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (SYNTHIA): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if Rio:
            rio_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, "Rio",
                                  crop_size=input_size, ignore_label=args.ignore_label,
                                  set='test', num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(rio_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Rio): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if Rome:
            rome_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, "Rome",
                                  crop_size=input_size, ignore_label=args.ignore_label,
                                  set='test', num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(rome_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Rome): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if Taipei:
            taipei_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, "Taipei",
                                  crop_size=input_size, ignore_label=args.ignore_label,
                                  set='test', num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(taipei_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Taipei): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

        if Tokyo:
            tokyo_loader = torch.utils.data.DataLoader(
                CrossCityDataSet(args.data_dir_target, "Tokyo",
                                  crop_size=input_size, ignore_label=args.ignore_label,
                                  set='test', num_classes=args.num_classes),
                batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

            hist = np.zeros((args.num_classes, args.num_classes))
            for i, data in enumerate(tokyo_loader):
                images_val, labels, _ = data
                images_val, labels = images_val.to(device), labels.to(device)
                pred = model(images_val, input_size)
                _, pred = pred.max(dim=1)

                labels = labels.cpu().numpy()
                pred = pred.cpu().detach().numpy()

                hist += fast_hist(labels.flatten(), pred.flatten(), args.num_classes)
            mIoUs = per_class_iu(hist)
            if args.mIoUs_per_class:
                for ind_class in range(args.num_classes):
                    print('==>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
            print('===> mIoU (Tokyo): ' + str(round(np.nanmean(mIoUs) * 100, 2)))
            print('=' * 50)

if __name__ == '__main__':
    main()
