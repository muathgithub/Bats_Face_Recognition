import os
import random
from os import makedirs
from os import listdir
from shutil import copyfile

# The number of bats from the same type in my case = 11
BATS_NUMBER = 11
# the ratio of the test images from all the images so the minimum number of the test images is 2 for each bat / bats type
videos_bats_test_ratio = 0.34
bats_by_type_ratio = 0.2
# random seed for getting the same split of data each time (0)
random.seed(0)

# the scr images directories
bats_by_types_dir = '../Bats_By_Types'
videos_bats_dir = '../Videos_Bats_Images'
black_out_dir = './black_out/r_u_q'

# the bats names/types labels
bats_by_type_labels = ['pteropus_niger', 'pteropus_poliocephalus', 'myotis_lucifugus', 'desmodus_rotundus', 'pteropus_medius']
videos_bats_labels = [f'bat_{i}' for i in range(BATS_NUMBER)]

# setting up the test_ratio and the labels_dirs for the needed case the same type bats / multiple types bats
# and setting the src directory of the images
test_ratio = bats_by_type_ratio
labels_dirs = bats_by_type_labels
src_directory = bats_by_types_dir

# create directories to devide the images bettween as directories tree
dataset_home = 'dataset_splits/'
sub_dirs = ['train/', 'test/']

if not os.path.exists(dataset_home):
    os.makedirs(dataset_home)

for sub_dir in sub_dirs:
    # create label subdirectories
    for label_dir in labels_dirs:
        new_dir = dataset_home + sub_dir + label_dir
        makedirs(new_dir, exist_ok=True)


# looping through th directories in the src directory
# and for each directory/label counting the images number
# in order to split them to train and test sets according to the chosen test ration
for directory in listdir(src_directory):

    images_num = 0
    curr_dir_path = os.path.join(src_directory, directory)

    # checking if the file is directory and if not continue
    if not os.path.isdir(curr_dir_path):
        continue

    # counting the images number for the train / test split
    for file in listdir(curr_dir_path):
        # checking if the file starts with . if true then continue (like .DS_Store in mac for configurations)
        if file.startswith("."):
            continue

        images_num += 1

    # choosing random images indexes for the test set
    random_images_num = int(test_ratio * images_num)
    random_images_indexes = random.sample(range(0, images_num), random_images_num)
    random_images_pointer = 0

    for file in listdir(curr_dir_path):
        if file.startswith("."):
            continue

        curr_image_path = os.path.join(curr_dir_path, file)

        # if the image index in the test random images indexes copy it to the test set directory
        if random_images_pointer in random_images_indexes:
            dst = os.path.join(dataset_home, 'test', directory, file)
            copyfile(curr_image_path, dst)

        else:
            dst = os.path.join(dataset_home, 'train', directory, file)
            copyfile(curr_image_path, dst)

        random_images_pointer += 1
