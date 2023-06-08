import time
import tensorflow as tf
from keras.saving.save import load_model
from tensorflow import keras
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.layers import Conv2D, UpSampling2D, BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import warnings

# These warnings ignore is for hiding the warnings from the console for clean observation
# like the warning of new versions of specific function in the matplotlib you can make it a comment to see these warnings
warnings.filterwarnings("ignore")

# variables for deciding when to start saving the models and to compare
# the current loss with the previous loss in the training
min_save_loss = 0.01
min_loss = 1e10


# defining the vgg16 auto encoder using the pretrained model of vgg16
def define_model(train_conv_layers):
    # getting the pretrained vgg16 model from keras
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # converting the layers to not trainable layers according to the
    # given boolean train_conv_layers, the default is trainable
    if not train_conv_layers:
        for layer in vgg16.layers:
            layer.trainable = False

    # Create the encoder part of the autoencoder using the pre-trained VGG16 model
    encoder = Model(inputs=vgg16.input, outputs=vgg16.get_layer('block5_conv3').output)

    # building the decoder by reversing the architecture of the vgg16 convolutional net
    decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(encoder.layers[-1].output)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2, 2))(decoder)

    decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(512, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2, 2))(decoder)

    decoder = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(256, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2, 2))(decoder)

    decoder = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(128, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = UpSampling2D((2, 2))(decoder)

    decoder = Conv2D(64, (3, 3), activation='relu', padding='same')(decoder)
    decoder = BatchNormalization()(decoder)
    decoder = Conv2D(3, (3, 3), activation='relu', padding='same')(decoder)

    autoencoder = Model(inputs=vgg16.input, outputs=decoder)

    return autoencoder


# plot diagnostic learning curves (train and test losses)
def summarize_diagnostics(history, val_loss, data_augmentation, train_conv_layers):
    fig, ax = plt.subplots()
    ax.plot(history['loss'], color='blue', label='train')
    ax.plot(history['val_loss'], color='orange', label='test')
    ax.set_title('Mean Squared Loss')

    # save plot to file
    plt.savefig(f"../Plots/Losses/VGG16_AE_{'%.6f' % val_loss}_plot.png")
    plt.close()


# this function trains the vgg16 autoencoder
def run_trainings(epochs_num, batch_size, data_augmentation, train_conv_layers):
    global min_loss
    # defining the model using the private function define_model()
    model = define_model(train_conv_layers)

    # Instantiate an optimizer to train the model.
    optimizer = Adam(learning_rate=0.001)

    # compiling the model with the training variables
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])

    # Instantiate a loss function.
    loss_fn = keras.losses.MeanSquaredError()

    # getting metrics object for calculating the MSE during the training of the model for each epoch
    train_loss_metric = keras.metrics.MeanSquaredError()
    val_loss_metric = keras.metrics.MeanSquaredError()

    # creating the data generator according to the given variable (data_augmentation)
    # with data augmentation or not
    if data_augmentation:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, featurewise_center=True,
                                           featurewise_std_normalization=True,
                                           rotation_range=90,
                                           width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
                                           vertical_flip=True)
    else:
        train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, featurewise_center=True,
                                           featurewise_std_normalization=True)

    # the test generator without data augmentation to see the true/clear result
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0, featurewise_center=True,
                                      featurewise_std_normalization=True)

    # prepare iterators for iterating through the images using the data generators
    train_it = train_datagen.flow_from_directory('../dataset_splits/train/',
                                                 class_mode='categorical', batch_size=batch_size,
                                                 target_size=(224, 224),
                                                 shuffle=True)

    test_it = test_datagen.flow_from_directory('../dataset_splits/test/',
                                               class_mode='categorical', batch_size=batch_size, target_size=(224, 224))

    # calculating the steps/batches number of the epoch
    train_steps_per_epoch = train_it.samples // batch_size + 1
    test_steps_per_epoch = test_it.samples // batch_size + 1
    # saving the losses after each epoch for the losses plots
    train_losses = []
    test_losses = []

    # training the model for epochs_num epochs
    for epoch in range(epochs_num):
        print(f"\nVGG16_AE Epoch {epoch}")
        # calculating the time for each epoch
        start_time = time.time()
        # variables for summing the batches losses in order to calculate the average epoch loss at the end of the epoch
        curr_epoch_train_losses_sum = 0
        curr_epoch_test_losses_sum = 0

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_it):
            # breaking the loop after going through all the samples/images (one epoch)
            if step >= train_steps_per_epoch:
                break

            # calculating the batch MSE loss by comparing the original train images and the reconstructed images
            with tf.GradientTape() as tape:
                train_reconstructed_images = model(x_batch_train, training=True)
                loss_value = loss_fn(x_batch_train, train_reconstructed_images)
                curr_epoch_train_losses_sum += loss_value

            # calculating the gradients and updating the weights
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_loss_metric.update_state(x_batch_train, train_reconstructed_images)

            # Thi code is for plotting one image result after each batch
            # f, ax = plt.subplots(nrows=1, ncols=2)
            # ax[0].imshow(x_batch_train[0])
            # ax[1].imshow(train_reconstructed_images[0])
            # plt.show()

        # calculating the final epoch train loss which is the average of all the epoch batches losses
        epoch_final_train_loss = float(curr_epoch_train_losses_sum / train_steps_per_epoch)
        train_losses.append(epoch_final_train_loss)
        print(f"Training loss: %.4f" % (epoch_final_train_loss,))

        # Reset training metrics at the end of each epoch for the next epoch
        train_loss_metric.reset_states()

        # Run a validation loop at the end of each epoch for calculating the validation loss
        for step, (x_batch_test, y_batch_test) in enumerate(test_it):
            if step >= test_steps_per_epoch:
                break

            # calculating the MSE loss by comparing the original test images and the reconstructed images
            test_reconstructed_images = model(x_batch_test, training=False)
            loss_value = loss_fn(x_batch_test, test_reconstructed_images)
            curr_epoch_test_losses_sum += loss_value

            # Update val metrics
            val_loss_metric.update_state(x_batch_test, test_reconstructed_images)

        # calculating the final epoch test loss which is the average of all the epoch batches losses
        epoch_final_test_loss = float(curr_epoch_test_losses_sum / test_steps_per_epoch)
        test_losses.append(epoch_final_test_loss)
        print(f"Validation loss: %.4f" % (epoch_final_test_loss,))

        print("Time taken: %.2fs" % (time.time() - start_time))

        # saving the model and the losses plot for it if there is an improvement in the results compared to the previous epoch
        # min_save_loss is a limit for starting saving the models because at the beginning the losses keep decreasing, but they are still high
        # so it's a waste of storage space to save all these model
        if epoch_final_train_loss < min_save_loss and epoch_final_train_loss < min_loss:
            min_loss = epoch_final_train_loss

            history = {"loss": train_losses, "val_loss": test_losses}
            # learning curves plot
            summarize_diagnostics(history, epoch_final_train_loss, data_augmentation, train_conv_layers)

            # save model
            model.save(f"../Models/VGG16_AE_{'%.6f' % epoch_final_test_loss}_{'%.6f' % epoch_final_train_loss}.h5")

        val_loss_metric.reset_states()


# This function uses a saved model, and it prints its results (original image / reconstructed image)
def plot_predictions():
    batch_size = 1

    model = load_model('../Models/VGG16_AE_0.010334_0.005259.h5')

    # declaring and initializing the image generator without data augmintation
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, featurewise_center=True,
                                 featurewise_std_normalization=True)

    # declaring and initializing the data iterator to iterate through the images in the directory
    data_it = datagen.flow_from_directory('../dataset_splits/test/',
                                          class_mode='categorical', batch_size=batch_size, target_size=(224, 224))

    # calculating the steps/batches number
    steps_num = data_it.samples // batch_size + 1

    # looping through the images and plotting the original and the reconstructed image
    for step, data in enumerate(data_it):
        if step >= steps_num:
            break

        images, labels = data

        reconstructed_images = model.predict(images)

        for i, img in enumerate(images):
            f, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(images[i])
            ax[1].imshow(reconstructed_images[i])
            plt.show()


if __name__ == "__main__":
    # epochs_num = int(1e10)
    # batch_size = 64
    #
    # run_trainings(epochs_num=epochs_num, batch_size=batch_size, data_augmentation=True, train_conv_layers=True)

    plot_predictions()
