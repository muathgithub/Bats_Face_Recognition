# this script divides the images of the testing set to the predicted directories
import os
from shutil import copyfile
import numpy as np
from keras.models import load_model
from keras_preprocessing import image

# create directories
main_dir = 'prediction_to_subs/'
sub_dirs = ['gray/', 'niger/']
for subdir in sub_dirs:
    new_dir = main_dir + subdir
    os.makedirs(new_dir, exist_ok=True)

model = load_model('../saved_models/VGG_0.982143.h5')

test_images_dir = 'test_set/'
test_images_number = len(os.listdir(test_images_dir))
image_size = 224
batch_holder = np.zeros((test_images_number, image_size, image_size, 3))


image_files_pathes = []

for i, curr_img_name in enumerate(os.listdir(test_images_dir)):
    image_files_pathes.append(curr_img_name)
    curr_img_PIL = image.load_img(os.path.join(test_images_dir, curr_img_name), target_size=(image_size, image_size))
    batch_holder[i, :] = curr_img_PIL

batch_holder = batch_holder.astype('float32')
batch_holder = batch_holder - [123.68, 116.779, 103.939]

result = model.predict(batch_holder)
# print(result)

for img_index, image_file_path in enumerate(image_files_pathes):
    src = test_images_dir + '/' + image_file_path
    if round(result[img_index, 0]) == 0:
        # print(f"Gray = {image_file_path}, {result[img_index, 0]}, {round(result[img_index, 0])}")
        dst = main_dir + '/' + sub_dirs[0] + image_file_path
        copyfile(src, dst)

    elif round(result[img_index, 0]) == 1:
        # print(f"Niger = {image_file_path}, {result[img_index, 0]}, {round(result[img_index, 0])}")
        dst = main_dir + '/' + sub_dirs[1] + image_file_path
        copyfile(src, dst)
