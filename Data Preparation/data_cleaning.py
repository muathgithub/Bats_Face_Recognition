import os
import numpy as np
import fiftyone as fo
import tensorflow as tf
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F
from sklearn.metrics.pairwise import cosine_similarity

# thresh hold to the similarity in order to decide if the image is duplicated or not
thresh = 0.85

# reding the data as directories tree
# dataset = fo.Dataset.from_dir(dataset_dir="dataset_other_vs_fruit_bats",
#                               dataset_type=fo.types.ImageClassificationDirectoryTree)

# reding data from one directory
dataset = fo.Dataset.from_images_dir("/Users/muathaj/Azrieli College/Fourth Year/Final_Project/DataSets/face_images/all_bats_faces")

# loading the pre_trained model to get the embeddings of the images
model = foz.load_zoo_model("inception-v3-imagenet-torch")  # inception-v3-imagenet-torch

# computing the embeddings of the images and converting them to np array
embeddings = dataset.compute_embeddings(model)
embeddings = np.asarray(embeddings)

# building similarity matrix between the images
similarity_matrix = cosine_similarity(embeddings)

# converting the images' similarity with themselves to zero
n = len(similarity_matrix)
similarity_matrix = similarity_matrix - np.identity(n)

# getting the ids of the images in the dataset
id_map = [s.id for s in dataset.select_fields(["id"])]

# adding similarity label to each image with the highest similarity between it and the other images
for idx, sample in enumerate(dataset):
    sample["max_similarity"] = similarity_matrix[idx].max()
    sample.save()

print("\ndataset.match: \n", dataset.match(F("max_similarity") > thresh))

# saving the ids of the images for both to remove and to keep
samples_to_remove = set()
samples_to_keep = set()

# looping through the hole dataset
for idx, sample in enumerate(dataset):
    if sample.id not in samples_to_remove:
        # Keep the first instance of two duplicates
        samples_to_keep.add(sample.id)

        dup_idxs = np.where(similarity_matrix[idx] > thresh)[0]
        for dup in dup_idxs:
            # we kept the first instance so remove all other duplicates
            samples_to_remove.add(id_map[dup])

        if len(dup_idxs) > 0:
            sample.tags.append("has_duplicates")
            sample.save()

    else:
        sample.tags.append("duplicate")
        sample.save()

print("samples_to_remove: {}, samples_to_keep: {}".format(len(samples_to_remove), len(samples_to_keep)))

# to remove the images that saved in the samples_to_remove from the file uncomment the remove for loop
# for pic_id in samples_to_remove:
#     os.remove(dataset[pic_id].filepath)


# adding the lables to the view and getting the new view
view = dataset.match_tags(["has_duplicates", "duplicate"])

# labelling each image with its duplication types in our case test / train
# it works if you get the data labeled (directories tree)
for idx, sample in enumerate(dataset):
    if sample.id in samples_to_remove:
        continue

    if sample.id in view:
        dup_idxs = np.where(similarity_matrix[idx] > thresh)[0]
        dup_splits = []
        dup_labels = {sample.ground_truth.label}
        for dup in dup_idxs:
            dup_sample = dataset[id_map[dup]]
            dup_split = "test" if "test" in dup_sample.tags else "train"
            dup_splits.append(dup_split)
            dup_labels.add(dup_sample.ground_truth.label)

        sample["dup_splits"] = dup_splits
        sample["dup_labels"] = list(dup_labels)
        sample.save()

train_w_test_dups = len(
    view.match(F("tags").contains("train")).match(F("dup_splits").contains("test"))
)

test_w_train_dups = len(
    view.match(F("tags").contains("test")).match(F("dup_splits").contains("train"))
)

label_mismatches = len(
    view.match(F("dup_labels").length() > 1)
)

print("liable mismatches: {}".format(label_mismatches / 2))

print("train_w_test_dups: {}, test_w_train_dups: {}".format(train_w_test_dups, test_w_train_dups))

print("Fob Compute:\n", fob.compute_uniqueness(dataset))

# adding filter to the view
max_similarity_view = dataset.exists("max_similarity").sort_by("max_similarity", reverse=True)

print("max_similarity\n", max_similarity_view)

session = fo.launch_app(dataset)

session.view = max_similarity_view

session.wait()
