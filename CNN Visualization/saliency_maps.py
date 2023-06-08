import os
import numpy as np
from keras.saving.save import load_model
from tensorflow import keras
from matplotlib import pyplot as plt
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from keras.preprocessing.image import ImageDataGenerator
from tf_keras_vis.scorecam import Scorecam
from matplotlib import cm
import warnings

# These warnings ignore is for hiding the warnings from the console for clean observation
# like the warning of new versions of specific function in the matplotlib you can make it a comment to see these warnings
warnings.filterwarnings("ignore")

videos_bats_labels_dict = {'bat_0': 0, 'bat_1': 1, 'bat_2': 2, 'bat_3': 3, 'bat_4': 4, 'bat_5': 5, 'bat_6': 6,
                           'bat_7': 7, 'bat_8': 8, 'bat_9': 9, 'bat_10': 10}

bats_by_type_labels_dict = {'pteropus_niger': 0, 'pteropus_poliocephalus': 1,
                            'myotis_lucifugus': 2, 'desmodus_rotundus': 3,
                            'pteropus_medius': 4}

classes_labels_dict = videos_bats_labels_dict


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


# This function returns the vanilla saliency map for the given images according to the given model
def vanilla(model, replace2linear, scores, processed_images):
    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)

    saliency_map = saliency(scores, processed_images)

    return saliency_map


# This function returns the smoothGrad saliency map for the given images according to the given model
def smooth_grad(model, replace2linear, scores, processed_images):
    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=replace2linear,
                        clone=True)

    # Generate saliency map with smoothing that reduce noise by adding noise
    saliency_map = saliency(scores,
                            processed_images,
                            smooth_samples=50,  # The number of calculating gradients iterations.
                            smooth_noise=0.20)  # noise spread level.

    saliency_map = normalize(saliency_map)

    return saliency_map


# This function returns the scoreCAM saliency map for the given images according to the given model
def score_CAM(model, replace2linear, scores, processed_images, image_size=(224, 224)):
    scorecam = Scorecam(model, model_modifier=replace2linear)

    # Generate heatmap with Faster-ScoreCAM
    cam = scorecam(scores,
                   processed_images,
                   penultimate_layer=-1,
                   max_N=10)

    ## Since v0.6.0, calling `normalize()` is NOT necessary.
    cam = normalize(cam)

    return cam


# This function build a plot that contains the bats faces images and the three with the saliency map result for each one of them
def all_in_one(model, net_name, replace2linear, image_size=(224, 224)):
    # getting the images data
    images_data, images_paths, images_scores, images_status, images_splits_names, images_probs, images_losses = get_images_data(
        model)
    images_data = np.asarray(images_data)

    done_bats = set()

    # looping through the images paths
    for image_path in images_paths:
        curr_bat_label = image_path.split("/")[-2]

        if curr_bat_label in done_bats:
            continue

        curr_bat_data_indexes = []

        # getting the indexes of the current bat images in the images' data lists
        for index, inner_image_path in enumerate(images_paths):
            if inner_image_path.split("/")[-2] == curr_bat_label:
                curr_bat_data_indexes.append(index)

        # creating the needed figure with the subplots
        f, ax = plt.subplots(nrows=images_num_for_bat, ncols=4, figsize=(25, 25), constrained_layout=True)
        f.subplots_adjust(left=None, bottom=0.5, right=None, top=0.8, wspace=0.01, hspace=0.6)
        f.suptitle(f"{curr_bat_label} [ {net_name} ]", fontsize=40)

        # for each bat image plot the image with the saliency map result for it
        for i, index in enumerate(curr_bat_data_indexes):
            # setting the curr image score in the proper object of the saliency maps library
            curr_bat_categorical_score = CategoricalScore([images_scores[index]])

            # getting the saliency maps of the current image
            curr_bat_cam_result = score_CAM(model, replace2linear, curr_bat_categorical_score,
                                            np.asarray([images_data[index]]))

            curr_bat_vanilla_result = vanilla(model, replace2linear, curr_bat_categorical_score,
                                              np.asarray([images_data[index]]))

            curr_bat_smooth_grad_result = smooth_grad(model, replace2linear, curr_bat_categorical_score,
                                                      np.asarray([images_data[index]]))

            image = keras.preprocessing.image.load_img(images_paths[index], target_size=image_size)

            # plotting the saliency maps of the current image
            ax[i][0].set_title(
                f"{images_paths[index].split('/')[-1]} | {images_splits_names[index]} | {images_status[index]} (%{int(images_probs[index] * 100)}) | loss: {images_losses[index]:.4f}",
                fontsize=20, pad=10)
            ax[i][0].imshow(image)
            heatmap = np.uint8(cm.jet(curr_bat_cam_result[0])[..., :3] * 255)
            ax[i][1].imshow(image)
            ax[i][1].imshow(heatmap, cmap='jet', alpha=0.4)
            ax[i][2].imshow(curr_bat_vanilla_result[0])
            ax[i][3].imshow(curr_bat_smooth_grad_result[0])

        # save the figure
        plt.savefig(f'all_in_one/{curr_bat_label}_{net_name}.png')
        done_bats.add(curr_bat_label)


# This function builds plots that contain the bats faces images and the three saliency map result for each one of them each plot type in a seperated figure
def separated_pictures(model, net_name, replace2linear, image_size=(224, 224)):
    # getting the images data
    images_data, images_paths, images_scores, images_status, images_splits_names, images_probs, images_losses = get_images_data(
        model)
    images_data = np.asarray(images_data)

    done_bats = set()

    # the plot figures types
    plots_types = ['Original', 'Vanilla', 'SmoothGrad', 'FastScoreCAM']

    # looping through the images paths
    for image_path in images_paths:
        curr_bat_label = image_path.split("/")[-2]

        if curr_bat_label in done_bats:
            continue

        # where to save the plots (path)
        curr_bat_plots_dir = f'all_in_one/final/ResNet50/{curr_bat_label}'

        # creating the directories according to the path if they are not exist
        if not os.path.exists(curr_bat_plots_dir):
            os.makedirs(curr_bat_plots_dir)

        curr_bat_data_indexes = []
        # dictionary for saving the sets names and the images number in each of them for each label
        images_sets_dict = {'train': 0, 'test': 0}

        # getting the indexes of the current bat images in the images' data lists
        for index, inner_image_path in enumerate(images_paths):
            if inner_image_path.split("/")[-2] == curr_bat_label:
                images_sets_dict[images_splits_names[index]] += 1
                curr_bat_data_indexes.append(index)

        # looping through the images sets train/test
        for images_set_name, images_num in images_sets_dict.items():

            # if the images number is larger than 36 then the rows and columns number = 6 -> 36 images
            if images_num >= max_figure_images:
                columns_num = rows_num = 6
            # else calculate the number of the rows and columns, so they fit in proper and nice to see arrangement
            else:
                columns_num = int(np.floor(np.sqrt(images_num)))
                rows_num = int(np.ceil(images_num / columns_num))

            # creating the needed figure with the subplots
            f, ax = plt.subplots(nrows=rows_num, ncols=columns_num, figsize=(25, 25), constrained_layout=True, squeeze=False)
            f.subplots_adjust(left=None, bottom=0.5, right=None, top=0.8, wspace=0.01, hspace=0.6)

            # for each plot type loop through all the images and plot them according to the plot type
            for plot_type in plots_types:

                f.suptitle(f"{curr_bat_label} [ {net_name} ] [ {plot_type} ] [ {images_set_name} set ]", fontsize=30)

                set_image_index = 0

                for i, index in enumerate(curr_bat_data_indexes):
                    # if we reached the images' max limit which is 36 images in the figure then break
                    if set_image_index >= max_figure_images:
                        break

                    if not images_splits_names[index] == images_set_name:
                        continue

                    curr_bat_categorical_score = CategoricalScore([images_scores[index]])

                    # setting the image data and model results in its subplot title
                    ax[set_image_index % rows_num][int(set_image_index / rows_num)].set_title(
                        f"{images_paths[index].split('/')[-1]} | {images_splits_names[index]} | {images_status[index]} (%{int(images_probs[index] * 100)}) | loss: {images_losses[index]:.4f}",
                        fontsize=9, pad=10)

                    # generate the proper plot according to the plot type original image or saliency map
                    if plot_type == 'Original':
                        image = keras.preprocessing.image.load_img(images_paths[index], target_size=image_size)

                        ax[set_image_index % rows_num][int(set_image_index / rows_num)].imshow(image)

                    elif plot_type == 'Vanilla':
                        curr_bat_vanilla_result = vanilla(model, replace2linear, curr_bat_categorical_score,
                                                          np.asarray([images_data[index]]))
                        ax[set_image_index % rows_num][int(set_image_index / rows_num)].imshow(
                            curr_bat_vanilla_result[0])

                    elif plot_type == 'SmoothGrad':
                        curr_bat_smooth_grad_result = smooth_grad(model, replace2linear, curr_bat_categorical_score,
                                                                  np.asarray([images_data[index]]))
                        ax[set_image_index % rows_num][int(set_image_index / rows_num)].imshow(
                            curr_bat_smooth_grad_result[0])

                    elif plot_type == 'FastScoreCAM':

                        image = keras.preprocessing.image.load_img(images_paths[index], target_size=image_size)

                        curr_bat_cam_result = score_CAM(model, replace2linear, curr_bat_categorical_score,
                                                        np.asarray([images_data[index]]))

                        heatmap = np.uint8(cm.jet(curr_bat_cam_result[0])[..., :3] * 255)
                        ax[set_image_index % rows_num][int(set_image_index / rows_num)].imshow(image)
                        ax[set_image_index % rows_num][int(set_image_index / rows_num)].imshow(heatmap, cmap='jet',
                                                                                               alpha=0.4)

                    set_image_index += 1

                # save the figure
                plt.savefig(f'{curr_bat_plots_dir}/{images_set_name}_{plot_type}_{curr_bat_label}_{net_name}.png')

        done_bats.add(curr_bat_label)


# This function return the images data for the train and the test sets
def get_images_data(model, batch_size=128):
    images_data = []
    images_scores = []
    images_status = []
    images_paths = []
    images_probs = []
    images_losses = []
    images_splits_names = []
    loss_fn = keras.losses.CategoricalCrossentropy()

    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    # specify imagenet mean values for centering
    datagen.mean = [0.485, 0.456, 0.406]

    images_splits_paths = ['dataset_splits/train/', 'dataset_splits/test/']

    for images_split_path in images_splits_paths:

        # prepare iterator (data generators)
        data_it = datagen.flow_from_directory(images_split_path,
                                              class_mode='categorical', batch_size=batch_size,
                                              target_size=(224, 224),
                                              shuffle=False, classes=classes_labels_dict)

        steps_per_epoch = data_it.samples // batch_size + 1

        images_paths = images_paths + [os.path.join(images_split_path, file_name) for file_name in data_it.filenames]

        split_name = images_split_path.split('/')[-2]
        images_splits_names = images_splits_names + [split_name for _ in range(data_it.samples)]

        # Iterate over the batches of the dataset.
        for step, (x_batch, y_batch) in enumerate(data_it):
            if step >= steps_per_epoch:
                break

            logits = model.predict(x_batch)

            images_data = images_data + list(x_batch)
            images_scores = images_scores + list(np.argmax(y_batch, axis=1))
            images_status = images_status + list((np.argmax(y_batch, axis=1) == np.argmax(logits, axis=1)))
            images_probs = images_probs + [float(logits[i][true_prob_index]) for i, true_prob_index in
                                           enumerate(np.argmax(y_batch, axis=1))]
            images_losses = images_losses + [loss_fn([y_batch[i]], [logits[i]]).numpy() for i in range(len(y_batch))]

    return images_data, images_paths, images_scores, images_status, images_splits_names, images_probs, images_losses


images_num_for_bat = 6
max_figure_images = 36

model = load_model('../Final_Save/Single_Type/Models/ResNet50_r_u_q_True_False_0.818182.h5')
net_name = "ResNet50 (81.8%)"

replace2linear = ReplaceToLinear()

all_in_one(model, net_name, replace2linear, image_size=(224, 224))
# separated_pictures(model, net_name, replace2linear, image_size=(224, 224))
