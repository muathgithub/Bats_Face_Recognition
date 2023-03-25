import os
from os import listdir

import numpy as np
from keras.saving.save import load_model
from tensorflow import keras
from matplotlib import pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear, ExtractIntermediateLayer, GuidedBackpropagation
from tf_keras_vis.utils.scores import CategoricalScore, InactiveScore
from keras.applications.resnet import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tf_keras_vis.scorecam import Scorecam
from matplotlib import cm
import warnings

# These warnings ignore is for hiding the warnings from the console for clean observation
# like the warning of new versions of specific function in the matplotlib you can make it a comment to see these warnings
warnings.filterwarnings("ignore")


def display_image(image_path, image_size=(224, 224)):
    # The local path to our target image
    img = keras.preprocessing.image.load_img(image_path, target_size=image_size)
    plt.imshow(img)
    plt.title(f"{image_path.split('/')[-1]}")
    plt.axis('off')
    plt.show()


def get_img_array(img_path, size=(224, 224)):
    # `img` is a PIL image of size 224x224
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (224, 224, 3)
    array = keras.preprocessing.image.img_to_array(img)

    return array


def smooth_grad(model, replace2linear, scores, processed_images):
    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(scores,
                            processed_images,
                            smooth_samples=100,  # The number of calculating gradients iterations.
                            smooth_noise=0.20)  # noise spread level.

    saliency_map = normalize(saliency_map)

    return saliency_map


def score_CAM(model, replace2linear, scores, image_paths, processed_images, image_size=(224, 224)):
    scorecam = Scorecam(model, model_modifier=replace2linear)

    # Generate heatmap with Faster-ScoreCAM
    cam = scorecam(scores,
                   processed_images,
                   penultimate_layer=-1,
                   max_N=10)

    ## Since v0.6.0, calling `normalize()` is NOT necessary.
    cam = normalize(cam)

    return cam


def all_in_one(model, net_name, replace2linear, image_size=(224, 224)):
    global processed_images

    images_data = []
    images_scores = []
    images_paths = []

    for directory in listdir(src_directory):

        curr_dir_path = os.path.join(src_directory, directory)

        if not os.path.isdir(curr_dir_path):
            continue

        for file in listdir(curr_dir_path):
            if not file.startswith("bat"):
                continue

            curr_image_path = os.path.join(curr_dir_path, file)

            images_data.append(get_img_array(curr_image_path))
            images_scores.append(int(directory.split("_")[1]))
            images_paths.append(curr_image_path)

        if net_name == 'ResNet50':
            processed_images = resnet50_preprocess_input(np.asarray(images_data))
        elif net_name == 'VGG16':
            processed_images = vgg16_preprocess_input(np.asarray(images_data))

    print(images_scores)
    print(images_paths)

    categorical_scores = CategoricalScore(images_scores)
    cam_results = score_CAM(model, replace2linear, categorical_scores, images_paths, processed_images)

    smooth_grad_results = smooth_grad(model, replace2linear, categorical_scores, processed_images)

    bats_num = 11
    images_num_for_bat = 6

    for j in range(bats_num):
        f, ax = plt.subplots(nrows=images_num_for_bat, ncols=3, figsize=(20, 20), constrained_layout=True)
        f.suptitle(f"bat_{images_scores[(j * images_num_for_bat)]}", fontsize=40)
        for i in range(images_num_for_bat):
            image = keras.preprocessing.image.load_img(images_paths[(j * images_num_for_bat) + i], target_size=image_size)
            ax[i][0].imshow(image)
            heatmap = np.uint8(cm.jet(cam_results[(j * images_num_for_bat) + i])[..., :3] * 255)
            ax[i][1].imshow(image)
            ax[i][1].imshow(heatmap, cmap='jet', alpha=0.5)
            ax[i][2].imshow(smooth_grad_results[(j * images_num_for_bat) + i])

        plt.savefig(f'all_in_one/bat_{images_scores[(j * images_num_for_bat)]}.png')


image_size = (224, 224)
# image_path = "./dataset_splits/train/bat_1/bat_1_6.jpg"

model = load_model('../Saved/ResNet50_f_True_False_1.000000.h5')
net_name = "ResNet50"

replace2linear = ReplaceToLinear()

src_directory = '../Bats_Images/'

all_in_one(model, net_name, replace2linear, image_size=(224, 224))

# score_CAM(model, replace2linear, score, image_path, processed_image)
# smooth_grad(model, replace2linear, score, processed_image)


#
# images_data = []
# images_scores = []
# images_paths = []
#
# curr_dir_path = "../Bats_Images/bat_4"
#
# for file in listdir(curr_dir_path):
#     if not file.startswith("bat"):
#         continue
#
#     curr_image_path = os.path.join(curr_dir_path, file)
#
#     images_data.append(get_img_array(curr_image_path))
#     images_scores.append(3)
#     images_paths.append(curr_image_path)
#
# categorical_scores = CategoricalScore(images_scores)
#
# processed_images = resnet50_preprocess_input(np.asarray(images_data))
#
# f, ax = plt.subplots(nrows=6, ncols=3, figsize=(20, 20), constrained_layout=True)
#
# for i in range(len(images_data)):
#     image = keras.preprocessing.image.load_img(images_paths[i], target_size=image_size)
#     ax[i][0].imshow(image)
#     categorical_scores = CategoricalScore(images_scores[i])
#     cam_results = score_CAM(model, replace2linear, categorical_scores, images_paths, processed_images)
#     heatmap = np.uint8(cm.jet(cam_results[0])[..., :3] * 255)
#     ax[i][1].imshow(image)
#     ax[i][1].imshow(heatmap, cmap='jet', alpha=0.5)
#     smooth_grad_results = smooth_grad(model, replace2linear, categorical_scores, processed_images)
#     ax[i][2].imshow(smooth_grad_results[0])
#
# plt.savefig(f'all_in_one/res_bat_4_test.png')
#
