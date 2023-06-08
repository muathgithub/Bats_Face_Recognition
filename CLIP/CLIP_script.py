import os
import numpy as np
import pandas as pd
import json
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# dictionaries for the classes labels
bats_by_type_labels_dict = {'pteropus_niger': 0, 'pteropus_poliocephalus': 1,
                            'myotis_lucifugus': 2, 'desmodus_rotundus': 3,
                            'pteropus_medius': 4}

# This function returns the images data, paths, files names adn labels
def get_images_data(src_directory):
    images_paths = []
    images_data = []
    images_labels = []
    images_files_names = []

    # looping through the images directories (classes directories)
    for directory in os.listdir(src_directory):

        curr_dir_path = os.path.join(src_directory, directory)

        if not os.path.isdir(curr_dir_path):
            continue

        # for each image append the image data in proper lists
        for file in os.listdir(curr_dir_path):
            if file.startswith("."):
                continue

            curr_image_path = os.path.join(curr_dir_path, file)

            images_paths.append(curr_image_path)
            images_files_names.append(file)
            images_data.append(Image.open(curr_image_path))
            images_labels.append(file.rsplit("_", 1)[0])

    return images_paths, images_files_names, images_labels, images_data


# get the bats types and descriptions from the given bats description files
def get_types_descriptions(file_path):
    bats_types_lables = []
    bats_types_descriptions = []

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

        # append the key = class label and the value = classe description to the proper lists
        for key, value in data.items():
            bats_types_lables.append(key)
            bats_types_descriptions.append(value)

    return bats_types_lables, bats_types_descriptions


# Function to preprocess images and classes descriptions
def preprocess(images, texts):
    # Tokenize texts
    encoded_inputs = tokenizer(texts, truncation=True, padding="max_length", max_length=max_sequence_length, return_tensors="pt")
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    # Preprocess images
    inputs = processor(images=images, input_ids=input_ids, attention_mask=attention_mask, return_tensors="pt")
    inputs.to(device)

    return inputs, input_ids


# This function returns the accuracy and a dictionary key = classe label
# value = list of the images that incorrectly classified to this class,
# so I can use to print a report
def get_accuracy_and_wrongs_dict(probs_ndarray, bats_types_lables, images_labels, images_files_names):
    wrongs_dict = {bat_type: [] for bat_type in bats_types_lables}
    wrongs_counter = 0

    for image_index, image_probs in enumerate(probs_ndarray):


        max_prob_label = bats_types_lables[np.argmax(image_probs)]

        if not images_labels[image_index] == max_prob_label:
            wrongs_dict[max_prob_label].append(images_files_names[image_index])
            wrongs_counter += 1

    accuracy = (len(images_files_names) - wrongs_counter) / len(images_files_names)

    return accuracy, wrongs_dict

# This function plots a confusion matrix for the model predections
def print_confusion_matrix(probs_ndarray, bats_types_lables, images_labels):

    # the predicted images labels according to the global dictionary labeling
    images_predicted_labels_by_dict = []
    for image_index, image_probs in enumerate(probs_ndarray):
        images_predicted_labels_by_dict.append(bats_by_type_labels_dict[bats_types_lables[np.argmax(image_probs)]])

    # the actual images labels according to the global dictionary labeling
    actual_images_labels_by_dict = [bats_by_type_labels_dict[image_label] for image_label in images_labels]

    # creating and plotting the confusion matrix
    cm = confusion_matrix(actual_images_labels_by_dict, images_predicted_labels_by_dict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    bats_types_lables, bats_types_descriptions = get_types_descriptions("../Bats_Types_Descriptions_1.json")

    images_paths, images_files_names, images_labels, images_data = get_images_data('../Bats_By_Types')

    # Set device
    device = "mps" if torch.has_mps else "cpu"

    # Set maximum sequence length
    max_sequence_length = 77

    # Load CLIP tokenizer
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    # Move the model to the same device as inputs
    model = model.to(device)

    inputs, input_ids = preprocess(images_data, bats_types_descriptions)

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs, input_ids=input_ids)

    # Access logits or probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    accuracy, wrongs_dict = get_accuracy_and_wrongs_dict(probs.cpu().numpy(), bats_types_lables, images_labels, images_files_names)

    print_confusion_matrix(probs.cpu().numpy(), bats_types_lables, images_labels)

    print(f"Accuracy = {accuracy}")
