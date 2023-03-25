import os
import random
import warnings
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from keras.applications import VGG16, ResNet50
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import plotly.express as px
from Unsupervised.VGG_AE.VGG_AE import AutoEncoderVGG, ImageFolderWithPaths

# These warnings ignore is for hiding the warnings from the console for clean observation
# like the warning of new versions of specific function in the matplotlib you can make it a comment to see these warnings
warnings.filterwarnings("ignore")

# directory where images are stored
from sklearn.utils import shuffle

Bats_Images_Dir = "../Bats_Images"


# This function reads and stores a stats about the dataset like the classes names, images count fo each class,
# images folder for the class and the images files names, and it stores them in panda framework
def dataset_stats():
    stats = []

    for (index, walk_result) in enumerate(os.walk(Bats_Images_Dir)):
        if index == 0:
            continue

        images_names = [image_name for image_name in walk_result[2] if image_name.startswith("bat")]
        images_names_count = len(images_names)
        sub_directory_name = walk_result[0].split("/")[-1]

        stats.append({"Code": sub_directory_name,
                      "Images count": images_names_count,
                      "Folder name": sub_directory_name,
                      "Files names": images_names})

    df = pd.DataFrame(stats)

    return df


# This function plots two random images from each class in the dataset
# in order to check that the program reads the data properly as I planned
def show_random_images(images, labels, number_of_images_to_show=2):
    # reversing the normalizing of the images, so we can see them in the original colors ranges
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    for code in list(set(labels)):

        same_code_images_indexes = [i for i, label in enumerate(labels) if label == code]

        random_indexes = set()
        while len(random_indexes) < number_of_images_to_show:
            random_indexes.add(random.choice(same_code_images_indexes))
        random_indexes = list(random_indexes)

        figure, axis = plt.subplots(1, number_of_images_to_show)
        figure.suptitle("{} random images for code {}".format(number_of_images_to_show, code), fontsize=20)

        for j in range(number_of_images_to_show):
            axis[j].imshow(inv_normalize(images[random_indexes[j]]).permute(1, 2, 0).abs())

        plt.show()


# loading the images and applying to them the needed preparation for training the model
# like resizing, normalizing and converting them to tensors datatype
def load_and_prepare_images():
    data_dir = "../Bats_Images"
    batch_size = 16
    images_tensors = []
    images_labels = []
    images_files_names = []

    transform = torchvision.transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])

    dataset = ImageFolderWithPaths(data_dir, transform)  # our custom dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for images, _, paths in data_loader:

        for i in range(len(paths)):
            images_tensors.append(images[i])
            images_labels.append(paths[i].split("/")[-2])
            images_files_names.append(paths[i].split("/")[-1])

    return images_tensors, images_labels, images_files_names


# This function receives a model and a data samples (images data) and it returns the convolutional nets
# predictions of these samples in out case the prediction is the flattened victor of the last convolutional layer
# in the given model before the fully connected layers
def covnet_transform(covnet_model, images_numpy_ndarray):
    # Pass our training data through the network
    pred = covnet_model.predict(images_numpy_ndarray)
    # Flatten the array
    flat = pred.reshape(images_numpy_ndarray.shape[0], -1)
    return flat


# Function that creates a PCA instance, fits it to the data and returns the instance
# in order to reduce the size of the features that represent each sample, so we get easier
# computations and save more memory
def to_PCA_features(data, n_components=None):
    p = PCA(n_components=n_components, random_state=0)
    return p.fit_transform(data)


# Function to plot the cumulative explained variance of PCA components
# This will help us decide how many components we should reduce
def pca_cumsum_plot(pca):
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


# This function creates a k-means model and fits it to the given data,
# so we can use it for future predictions/clustering
def create_train_kmeans(data, number_of_clusters):
    k = KMeans(n_clusters=number_of_clusters, random_state=0)

    # start = time.time()
    k.fit(data)
    # end = time.time()
    # print("Training took {} seconds".format(end - start))

    return k


# This function creates a hierarchical model and fits it to the given data,
# so we can use it for future predictions/clustering
def create_train_hierarchical(data, number_of_clusters):
    h = AgglomerativeClustering(n_clusters=number_of_clusters, affinity='euclidean', linkage='ward')

    # start = time.time()
    h.fit(data)
    # end = time.time()
    # print("Training took {} seconds".format(end - start))

    return h


# This function counts how many images from each class there is in each cluster
# which we need for calculating the accuracy of the clustering algorithms
def cluster_label_count(clusters, labels):
    count = {}

    # Get unique clusters and labels
    unique_clusters = list(set(clusters))
    unique_labels = list(set(labels))

    # Create counter for each cluster/label combination and set it to 0
    for cluster in unique_clusters:
        count[cluster] = {}

        for label in unique_labels:
            count[cluster][label] = 0

    # Let's count
    for i in range(len(clusters)):
        count[clusters[i]][labels[i]] += 1

    cluster_df = pd.DataFrame(count)

    return cluster_df


# This function returns the scores of the given data predictions
# in our case we just return the accuracy
def print_scores(true, pred):
    acc = accuracy_score(true, pred)
    return "\n\tAccuracy: {0:0.8f}".format(acc)


# This function displays a 3d tsne for the given data samples with there files names
# which helps in observing the data to understand the efficiency of the models and
# the samples that the models fail to predict good
def display_3d_tsne(data, y_train, files_names):
    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(data, )

    fig = px.scatter_3d(
        projections, x=0, y=1, z=2,
        color=y_train, labels={'color': 'Bat'}, hover_name=files_names
    )
    fig.update_traces(marker_size=6)
    fig.show()


# This function displays a 2d tsne for the given data samples with there files names
# which helps in observing the data to understand the efficiency of the models and
# # the samples that the models fail to predict good
def display_2d_tsne(data, y_train, files_names):
    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(data)

    fig = px.scatter(
        projections, x=0, y=1,
        color=y_train, labels={'color': 'Bat'}, hover_name=files_names
    )

    fig.update_traces(marker_size=6)
    fig.show()


# This function creates, trains and runs kmeans algorithm on the given data,
# and it prints the accuracy of the model using manually declarations of the clusters labels
# ofcourse it uses multiple function to get all these jobs done
def run_kmeans(data, labels, number_of_clusters, net_name):
    kmeans_model = create_train_kmeans(data, number_of_clusters)
    kmeans_pred = kmeans_model.predict(data)
    # kmeans_cluster_count = cluster_label_count(kmeans_pred, labels)
    # print(f"Kmeans {net_name}")
    # print(kmeans_cluster_count)
    cluster_code = []

    if net_name == "VGG16":
        # Manually adjust these lists so that the index of each label reflects which cluter it lies in
        cluster_code = ["bat_10", "bat_4", "bat_5", "bat_8", "bat_6", "bat_1", "bat_3", "bat_11", "bat_9", "bat_7",
                        "bat_2"]

    elif net_name == "ResNet50":
        cluster_code = ["bat_4", "bat_8", "bat_10", "bat_5", "bat_3", "bat_6", "bat_7", "bat_11", "bat_1", "bat_9",
                        "bat_2"]

    elif net_name == "VGG_AE":
        cluster_code = ["bat_6", "bat_5", "bat_2", "bat_11", "bat_3", "bat_9", "bat_4", "bat_7", "bat_1", "bat_10",
                        "bat_8"]  # 6->0 5->1 2->2 11->3 3->4 9->5 4->6 7->7 1->8 10->9 8->10

    pred_codes = [cluster_code[x] for x in kmeans_pred]

    print(f"KMeans {net_name}", print_scores(labels, pred_codes))


# This function creates, trains and runs hierarchical algorithm on the given data,
# and it prints the accuracy of the model using manually declarations of the clusters labels
# ofcourse it uses multiple function to get all these jobs done
def run_hierarchical(data, labels, number_of_clusters, net_name):
    hierarchical_model = create_train_hierarchical(data, number_of_clusters)
    hierarchical_pred = hierarchical_model.labels_
    hierarchical_cluster_count = cluster_label_count(hierarchical_pred, labels)
    print(f"Hierarchical {net_name}")
    print(hierarchical_cluster_count)
    cluster_code = []

    if net_name == "VGG16":
        # Manually adjust these lists so that the index of each label reflects which cluter it lies in
        cluster_code = ["bat_11", "bat_10", "bat_2", "bat_7", "bat_5", "bat_6", "bat_8", "bat_3", "bat_4", "bat_1",
                        "bat_9"]  # 11-0 10-1 2-2 7-3 5-4 6-5 8-6 3-7 4-8 1-9 9-10

    elif net_name == "ResNet50":
        cluster_code = ["bat_1", "bat_7", "bat_8", "bat_2", "bat_9", "bat_11", "bat_6", "bat_10", "bat_3", "bat_5",
                        "bat_4"]  # 1->0 7->1 8->2 2->3 9->4 11->5 6->6 10->7 3->8 5->9 4->10

    elif net_name == "VGG_AE":
        cluster_code = ["bat_10", "bat_4", "bat_9", "bat_1", "bat_5", "bat_6", "bat_11", "bat_3", "bat_8", "bat_7",
                        "bat_2"]  # 10->0 4->1 9->2 1->3 5->4 6->5 11->6 3->7 8->8 7->9 2->10

    pred_codes = [cluster_code[x] for x in hierarchical_pred]

    print(f"Hierarchical {net_name}", print_scores(labels, pred_codes))


# this function returns the best option to run the code in GPU/MPS/CPU
def get_device():
    if torch.cuda.is_available():
        device = 'cuda'

    # 'mps' this option has to be 'mps' which is the gpu of the new Apple Silicon but the
    # UnPooling layer in pytorch still not working on the 'mps' so I wrote 'cpu' instead
    elif torch.has_mps:
        device = 'cpu'  # 'mps'
    else:
        device = 'cpu'

    return device


dataset = dataset_stats().set_index("Code")
codes = dataset.index.tolist()
number_of_clusters = len(codes)

# loading the images for training and prediction
images_tensors, images_labels, images_files_names = load_and_prepare_images()
# converting the images from tensors to ndarrays which is the data type that alot of models work with
images_numpy_ndarray = np.array([image_tensor.permute(1, 2, 0).numpy() for image_tensor in images_tensors])
# show_random_images(images_tensors, images_labels)

# Load the models with ImageNet weights
vgg16_model = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
resnet50_model = ResNet50(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

# training the models on the dataset and getting there predictions
vgg16_output = covnet_transform(vgg16_model, images_numpy_ndarray)
print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))
# applying the PCA model on the outputs of the models to reduce the features victors size
vgg16_output_pca = to_PCA_features(vgg16_output)

# training the models on the dataset and getting there predictions
resnet50_output = covnet_transform(resnet50_model, images_numpy_ndarray)
print("ResNet50 flattened output has {} features".format(resnet50_output.shape[1]))
# applying the PCA model on the outputs of the models to reduce the features victors size
resnet50_output_pca = to_PCA_features(resnet50_output)

# running kmeans and hierarchical on vgg16 outputs
run_kmeans(vgg16_output_pca, images_labels, number_of_clusters, "VGG16")
run_hierarchical(vgg16_output_pca, images_labels, number_of_clusters, "VGG16")

print()

# running kmeans and hierarchical on resnet50 outputs
run_kmeans(resnet50_output_pca, images_labels, number_of_clusters, "ResNet50")
run_hierarchical(resnet50_output_pca, images_labels, number_of_clusters, "ResNet50")

######################################## VGG_AE ########################################

# This is an autoencoder model that custom-built on VGG net architecture
# The model have been trained on 1271 images of bats faces for 1322 epochs the final MSELoss is 0.0029086507856845856

device = torch.device(get_device())
model = AutoEncoderVGG(device)
# loading the trained model on the cpu (because the UnPooling layer doesn't work on the Apple Silicon MSP [GPU])
model.load_state_dict(torch.load("Models/model_1322.pt", map_location=torch.device('cpu')))
images_flattend_codes = []

# getting the codes of the image from the encoder of the vgg autoencoder
for image_tensor in images_tensors:
    curr_image_code_tensor, _ = model.encoder.forward(torch.unsqueeze(image_tensor, dim=0))
    images_flattend_codes.append(torch.flatten(curr_image_code_tensor.detach()))

# converting the images from tensors to ndarrays which is the data type that alot of models work with
flattend_codes_numpy = np.array([flattend_code.numpy() for flattend_code in images_flattend_codes])
# applying the PCA model on the outputs of the models to reduce the features victors size
flattend_codes_pca = to_PCA_features(flattend_codes_numpy)

print()

# running kmeans and hierarchical on vgg encoder outputs
run_kmeans(flattend_codes_pca, images_labels, number_of_clusters, "VGG_AE")
run_hierarchical(flattend_codes_pca, images_labels, number_of_clusters, "VGG_AE")

# displaying the net outputs using3d tsne
# display_3d_tsne(vgg16_output_pca, images_labels, images_files_names)
# display_3d_tsne(resnet50_output_pca, images_labels, images_files_names)
# display_3d_tsne(flattend_codes_pca, images_labels, images_files_names)
