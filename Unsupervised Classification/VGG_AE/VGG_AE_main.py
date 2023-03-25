import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from VGG_AE import AutoEncoderVGG
import warnings

warnings.filterwarnings("ignore")
min_loss = 10e6
last_model_name = None
last_ae_comp_name = None
last_losses_name = None


# This function trains the autoencoder.It reads the images applies the transformations
# on the image resizing, normalization and converting to tensor then it trains the model on these images
# with the ability of plotting losses graph and images samples to observe the training process,
# and it saves the model with the minimal loss
def train(model, epochs_num, device):
    global last_model_name, min_loss
    batch_size = 16
    # limits for when to save or/and display the samples plots
    ae_comp_save_display_mod = 2
    # limit for when to start saving the model (after which epochs number)
    start_saving_models_lim = 2

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                    std=[0.229, 0.224, 0.225])])
    training_set = torchvision.datasets.ImageFolder('../../Videos_Labelbox', transform=transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    epochs_losses_list = []

    i, inputs, outputs = None, None, None
    for epoch_num in range(epochs_num):

        print('EPOCH {}:'.format(epoch_num + 1))
        curr_epoch_loss = 0

        for i, data in enumerate(training_loader):
            inputs, labels = data
            print(inputs)
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.to(device)

            loss = loss_fn(outputs, inputs)

            loss.backward()

            optimizer.step()

            batch_loss_val = loss.item()
            curr_epoch_loss += batch_loss_val

            print('  batch {} loss: {}'.format(i + 1, batch_loss_val))

        last_epoch_loss = curr_epoch_loss / i + 1
        epochs_losses_list.append(last_epoch_loss)

        # if (epoch_num + 1) % ae_comp_save_display_mod == 0:
        #     img_save_display(inputs[0], outputs[0], epoch_num + 1, save=True, display=True)

        if epoch_num + 1 > start_saving_models_lim and last_epoch_loss < min_loss:
            if last_model_name:  # removing the previous model
                os.remove(f'../Models/{last_model_name}')

            last_model_name = f'model_{epoch_num + 1}.pt'
            torch.save(model.state_dict(), f'../Models/{last_model_name}')
            plot_loss_values(epochs_losses_list, epoch_num + 1, save=True, display=True)
            min_loss = last_epoch_loss
            img_save_display(inputs[0], outputs[0], epoch_num + 1, save=True, display=True)

    return epochs_losses_list, epochs_num


# This function saves or/and displays sample images for the observation of the model training or results
def img_save_display(original_img, reconstructed_img, epoch_num=-1, save=False, display=True):
    global last_ae_comp_name

    # reversing the normalization of the image in order to see it in the original colors
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    original_img = inv_normalize(original_img.detach()).permute(1, 2, 0).abs()
    reconstructed_img = inv_normalize(reconstructed_img.detach()).permute(1, 2, 0).abs()

    plt.clf()
    figure, axis = plt.subplots(1, 2)
    axis[0].imshow(original_img)
    axis[1].imshow(reconstructed_img)
    axis[0].axis('off')
    axis[1].axis('off')

    if save:
        if last_ae_comp_name:
            os.remove(f'../Plots/Comps/{last_ae_comp_name}')

        last_ae_comp_name = f'ae_comp_{epoch_num}.png'
        plt.savefig(f'../Plots/Comps/{last_ae_comp_name}')

    if display:
        plt.show()


# this function saves or/and displays the loss function given the losses values lists and the epochs number
def plot_loss_values(epochs_losses_list, epoch_num=-1, save=False, display=True):
    global last_losses_name

    plt.clf()
    plt.plot(range(0, epoch_num), epochs_losses_list, label='Training Loss')

    # Add in a title and axes labels
    plt.title('AE Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Set the tick locations
    plt.xticks(range(0, epoch_num, int(10 ** (float(np.log10(epoch_num)) - 1)) + 1))

    # Display the plot
    plt.legend(loc='best')

    if save:
        if last_losses_name:
            os.remove(f'../Plots/Losses/{last_losses_name}')

        last_losses_name = f'losses_{epoch_num}.png'
        plt.savefig(f'../Plots/Losses/{last_losses_name}')

    if display:
        plt.show()


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


epochs_num = int(10e9)

device = torch.device(get_device())
model = AutoEncoderVGG(device)

train(model, epochs_num, device)
#
# model.load_state_dict(torch.load("model_895.pt", map_location=torch.device('cpu')))
#
#
# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# transform = transforms.Compose(
#     [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                 std=[0.229, 0.224, 0.225])])
# training_set = torchvision.datasets.ImageFolder('../../Bats_Images', transform=transform)
# training_loader = DataLoader(training_set, batch_size=16, shuffle=True)
#
# loss_sum = 0
# for data in training_loader:
#
#     inputs, labels = data
#
#     inputs = inputs.to(device)
#
#     optimizer.zero_grad()
#
#     outputs = model(inputs)
#     outputs = outputs.to(device)
#
#     loss = loss_fn(outputs, inputs)
#
#     loss_sum += loss.item()
#
#
# print(loss_sum/66)


