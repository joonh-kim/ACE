import glob
import os

data_path = "/work/IDD_Segmentation/leftImg8bit"
sub_dir = "train"

txtfiles = []
os.path.join(data_path, sub_dir, '*', '*.png')
for file in glob.glob(os.path.join(data_path, sub_dir, '*', '*.png')):
    txtfiles.append(file.split("/")[-2] + '/' + file.split("/")[-1])

if not os.path.exists("./idd_list"):
    os.makedirs("./idd_list")

writefile = open(os.path.join('./idd_list/', sub_dir + ".txt"), 'w')
for file in txtfiles:
    writefile.writelines(file + '\n')
