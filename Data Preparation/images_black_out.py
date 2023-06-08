import os
import cv2
import numpy as np
from os import makedirs
from os import listdir

# The number of bats from the same type in my case = 11
BATS_NUMBER = 11
# the scr images directories
bats_by_types_dir = '../Bats_By_Types'
videos_bats_dir = '../Videos_Bats_Images'
black_out_dir = './black_out/r_h'

# the bats names/types labels
bats_by_type_labels = ['pteropus_niger', 'pteropus_poliocephalus', 'myotis_lucifugus', 'desmodus_rotundus', 'pteropus_medius']
videos_bats_labels = [f'bat_{i}' for i in range(BATS_NUMBER)]

# setting up the labels_dirs for the needed case the same type bats / multiple types bats
# and setting the src directory of the images
labels_dirs = videos_bats_labels
src_directory = videos_bats_dir

# The directories to save the images in
dataset_home = 'black_out/'
# all the black_out options
sub_dirs = ['r_h/', 'l_h/', 'r_u_q', 'l_u_q', 'r_b_q', 'l_b_q']

# looping through the black_out option and creating directory for each one
for sub_dir in sub_dirs:

    for label_dir in labels_dirs:
        new_dir = os.path.join(dataset_home, sub_dir, label_dir)
        makedirs(new_dir, exist_ok=True)

# generating the black_out images for each option
for i, sub_dir in enumerate(sub_dirs):

    # looping through the original image
    for directory in listdir(src_directory):

        curr_dir_path = os.path.join(src_directory, directory)

        if not os.path.isdir(curr_dir_path):
            continue

        # checking if the file starts with . if true then continue (like .DS_Store in mac for configurations)
        for file in listdir(curr_dir_path):
            if file.startswith("."):
                continue

            curr_image_path = os.path.join(curr_dir_path, file)

            # Load the image
            img = cv2.imread(curr_image_path)

            # Get the dimensions of the image
            height, width, _ = img.shape

            # all the black_out options according to the order of the black_out options directories
            black_out_area = [(0, height, width // 2, width), (0, height, 0, width // 2),
                              (0, height // 2, width // 2, width), (0, height // 2, 0, width // 2),
                              (height // 2, height, width // 2, width), (height // 2, height, 0, width // 2)]

            # choosing the black_out area dimensions according to the black_out option directory index in the directories names list (sub_dirs)
            sub_dir_area = black_out_area[i]

            # Create a numpy array of zeros with the same dimensions as the image
            mask = np.zeros((height, width), dtype=np.uint8)

            # setting the mask to white according to the black_out_area dimensions
            mask[sub_dir_area[0]:sub_dir_area[1], sub_dir_area[2]:sub_dir_area[3]] = 255

            # Apply the mask to the image
            img = cv2.bitwise_and(img, img, mask=mask)

            save_path = os.path.join(dataset_home, sub_dir, directory, file)

            # save the blacked out image
            cv2.imwrite(save_path, img)
