from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# create directories to devide the images bettween as directories tree
dataset_home = 'dataset_gray_vs_niger_bats/'
sub_dirs = ['train/', 'test/']
for sub_dir in sub_dirs:
    # create label subdirectories
    label_dirs = ['gray/', 'niger/']
    for label_dir in label_dirs:
        new_dir = dataset_home + sub_dir + label_dir
        makedirs(new_dir, exist_ok=True)


# seed for the random numbers' generator to get the same split each time
seed(1)

# define ratio of images to use for validation / test
val_ratio = 0.25

# copy the images to the proper directories from all_images directory
src_directory = 'gray_niger/'
for file in listdir(src_directory):
    src = src_directory + '/' + file
    dst_dir = 'train/'
    if random() < val_ratio:
        dst_dir = 'test/'
    if file.startswith('gray'):
        dst = dataset_home + dst_dir + 'gray/' + file
        copyfile(src, dst)
    elif file.startswith('niger'):
        dst = dataset_home + dst_dir + 'niger/' + file
        copyfile(src, dst)
