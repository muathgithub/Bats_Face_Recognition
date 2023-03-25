import os
import random
from os import makedirs
from os import listdir
from shutil import copyfile


BATS_NUMBER = 11
test_ratio = 0.34

random.seed(0)

# create directories to devide the images bettween as directories tree
dataset_home = 'dataset_splits/'
sub_dirs = ['train/', 'test/']
for sub_dir in sub_dirs:
    # create label subdirectories
    label_dirs = [f'bat_{i}' for i in range(BATS_NUMBER)]
    for label_dir in label_dirs:
        new_dir = dataset_home + sub_dir + label_dir
        makedirs(new_dir, exist_ok=True)


# copy the images to the proper directories from all_images directory
src_directory = '../Supervised Classification/black_out/r_h'
for directory in listdir(src_directory):

    images_num = 0
    curr_dir_path = os.path.join(src_directory, directory)

    if not os.path.isdir(curr_dir_path):
        continue

    for file in listdir(curr_dir_path):
        if not file.startswith("bat"):
            continue

        images_num += 1

    random_images_num = int(test_ratio * images_num)
    random_images_indexes = random.sample(range(0, images_num), random_images_num)
    random_images_pointer = 0

    for file in listdir(curr_dir_path):
        if not file.startswith("bat"):
            continue

        curr_image_path = os.path.join(curr_dir_path, file)

        if random_images_pointer in random_images_indexes:
            dst = os.path.join(dataset_home, 'test', directory, file)
            copyfile(curr_image_path, dst)

        else:
            dst = os.path.join(dataset_home, 'train', directory, file)
            copyfile(curr_image_path, dst)

        random_images_pointer += 1
