import os
import cv2
import numpy as np
from os import makedirs
from os import listdir

BATS_NUMBER = 11

dataset_home = 'black_out/'
sub_dirs = ['r_h/', 'l_h/', 'r_u_q', 'l_u_q', 'r_b_q', 'l_b_q']

for sub_dir in sub_dirs:
    # create label subdirectories
    label_dirs = [f'bat_{i}' for i in range(BATS_NUMBER)]
    for label_dir in label_dirs:
        new_dir = os.path.join(dataset_home, sub_dir, label_dir)
        makedirs(new_dir, exist_ok=True)

src_directory = '../Bats_Images/'
for i, sub_dir in enumerate(sub_dirs):
    for directory in listdir(src_directory):

        curr_dir_path = os.path.join(src_directory, directory)

        if not os.path.isdir(curr_dir_path):
            continue

        for file in listdir(curr_dir_path):
            if not file.startswith("bat"):
                continue

            curr_image_path = os.path.join(curr_dir_path, file)

            # Load the image
            img = cv2.imread(curr_image_path)

            # Get the dimensions of the image
            height, width, _ = img.shape

            black_out_area = [(0, height, width // 2, width), (0, height, 0, width // 2),
                              (0, height // 2, width // 2, width), (0, height // 2, 0, width // 2),
                              (height // 2, height, width // 2, width), (height // 2, height, 0, width // 2)]
            sub_dir_area = black_out_area[i]
            # Create a numpy array of zeros with the same dimensions as the image
            mask = np.zeros((height, width), dtype=np.uint8)

            mask[sub_dir_area[0]:sub_dir_area[1], sub_dir_area[2]:sub_dir_area[3]] = 255

            # Apply the mask to the image
            img = cv2.bitwise_and(img, img, mask=mask)

            save_path = os.path.join(dataset_home, sub_dir, directory, file)

            # print(save_path)
            # save the blacked out image
            cv2.imwrite(save_path, img)
