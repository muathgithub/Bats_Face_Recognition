import time
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense
from keras import regularizers
from keras.layers import Flatten
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.layers import Input, Dropout
from keras.saving.save import load_model
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# These warnings ignore is for hiding the warnings from the console for clean observation
# like the warning of new versions of specific function in the matplotlib you can make it a comment to see these warnings
warnings.filterwarnings("ignore")

# dictionaries for the classes labels and there presentation value in the nets
videos_bats_labels_dict = {'bat_0': 0, 'bat_1': 1, 'bat_2': 2, 'bat_3': 3, 'bat_4': 4, 'bat_5': 5, 'bat_6': 6,
                           'bat_7': 7, 'bat_8': 8, 'bat_9': 9, 'bat_10': 10}

bats_by_type_labels_dict = {'pteropus_niger': 0, 'pteropus_poliocephalus': 1,
                            'myotis_lucifugus': 2, 'desmodus_rotundus': 3,
                            'pteropus_medius': 4}

# These variables for saving the models best accuracy during the training
min_save_acc = 0.5
best_accuracy_dict = {"VGG16": -1, "ResNet50": -1}

# choosing the needed labels dictionary
classes_labels_dict = bats_by_type_labels_dict
CLASSES_NUMBER = len(classes_labels_dict)


# defining the pre_trained model with costume top (fully connected) according to the given net_name
def define_model(net_name, train_conv_layers):
    model = None

    input_tensor = Input(shape=(224, 224, 3))

    # getting the wanted pretrained model from keras
    if net_name == "VGG16":
        model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    elif net_name == "ResNet50":
        model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # converting the layers to not trainable layers according to the
    # given boolean train_conv_layers, the default is trainable
    if not train_conv_layers:
        for layer in model.layers:
            layer.trainable = False

    # add new classifier layers (fully connected layer with dropouts and penalties)
    top_model = Flatten()(model.layers[-1].output)

    top_model = Dense(4096, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                      bias_regularizer=regularizers.L2(1e-4),
                      activity_regularizer=regularizers.L2(1e-5))(top_model)

    top_model = Dropout(0.2)(top_model)

    top_model = Dense(1072, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                      bias_regularizer=regularizers.L2(1e-4),
                      activity_regularizer=regularizers.L2(1e-5))(top_model)

    top_model = Dropout(0.2)(top_model)

    output_layer = Dense(CLASSES_NUMBER, activation='softmax')(top_model)

    # define new model
    model = Model(inputs=model.inputs, outputs=output_layer)

    return model


# plot diagnostic learning curves (train and test losses and accuracies)
def summarize_diagnostics(history, acc, net_name, data_augmentation, train_conv_layers):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(history['loss'], color='blue', label='train')
    axs[0].plot(history['val_loss'], color='orange', label='test')
    axs[0].set_title('Cross Entropy Loss')

    axs[1].plot(history['accuracy'], color='blue', label='train')
    axs[1].plot(history['val_accuracy'], color='orange', label='test')
    axs[1].set_title('Classification Accuracy')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # save plot to file
    plt.savefig(f"Plots/Losses/{net_name}_{data_augmentation}_{train_conv_layers}_{'%.6f' % acc}_plot.png")
    plt.close()


def run_trainings(net_name, epochs_num, batch_size, data_augmentation, train_conv_layers):
    # defining the model using the private function define_model()
    model = define_model(net_name, train_conv_layers=train_conv_layers)

    # Instantiate an optimizer to train the model.
    optimizer = Adam(learning_rate=0.0001)

    # Instantiate an optimizer to train the model.
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # Instantiate a loss function.
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

    # getting metrics object for calculating the categorical loss during the training of the model for each epoch
    train_acc_metric = keras.metrics.CategoricalAccuracy()
    val_acc_metric = keras.metrics.CategoricalAccuracy()

    # creating the data generator according to the given variable (data_augmentation)
    # with data augmentation or not
    if data_augmentation:
        train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
                                           rotation_range=90,
                                           width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
                                           vertical_flip=True)
    else:
        train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    # the test generator without data augmentation to see the true/clear result
    test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

    # specify imagenet mean values for centering
    train_datagen.mean = [0.485, 0.456, 0.406]
    test_datagen.mean = [0.485, 0.456, 0.406]

    # prepare iterators for iterating through the images using the data generators
    train_it = train_datagen.flow_from_directory('dataset_splits/train/',
                                                 class_mode='categorical', batch_size=batch_size,
                                                 target_size=(224, 224),
                                                 shuffle=True, classes=classes_labels_dict)

    test_it = test_datagen.flow_from_directory('dataset_splits/test/',
                                               class_mode='categorical', batch_size=batch_size, target_size=(224, 224),
                                               classes=classes_labels_dict, shuffle=False)

    # Printing the classes labels numbers
    print(train_it.class_indices)

    # calculating the steps/batches number of the epoch
    train_steps_per_epoch = train_it.samples // batch_size + 1
    test_steps_per_epoch = test_it.samples // batch_size + 1

    # saving the losses and the accuracies after each epoch for the losses plots
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # training the model for epochs_num epochs
    for epoch in range(epochs_num):
        print(f"\n{net_name} Epoch {epoch}")
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

            # calculating the batch categorical loss
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
                curr_epoch_train_losses_sum += loss_value

            # calculating the gradients and updating the weights
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_batch_train, logits)

        # calculating the final epoch train loss which is the average of all the epoch batches losses
        epoch_final_train_loss = float(curr_epoch_train_losses_sum / train_steps_per_epoch)
        train_losses.append(epoch_final_train_loss)
        print(f"Training loss: %.4f" % (epoch_final_train_loss,))

        # Display the train accuracy metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        train_accuracies.append(train_acc)
        print("Training acc: %.4f" % (float(train_acc),))
        # Display the test accuracy using keras evaluation method.
        # _, keras_train_acc = model.evaluate(train_it)
        # print(f"Keras acc = {keras_train_acc}")

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, (x_batch_test, y_batch_test) in enumerate(test_it):
            if step >= test_steps_per_epoch:
                break

            # calculating the categorical loss for validation images
            test_logits = model(x_batch_test, training=False)
            loss_value = loss_fn(y_batch_test, test_logits)
            curr_epoch_test_losses_sum += loss_value

            # Update val metrics
            val_acc_metric.update_state(y_batch_test, test_logits)

        # calculating the final epoch teat loss which is the average of all the epoch batches losses
        epoch_final_test_loss = float(curr_epoch_test_losses_sum / test_steps_per_epoch)
        test_losses.append(epoch_final_test_loss)
        print(f"Validation loss: %.4f" % (epoch_final_test_loss,))

        # Display the test accuracy metrics at the end of each epoch.
        val_acc = val_acc_metric.result()
        test_accuracies.append(val_acc)
        print("Validation acc: %.4f" % (float(val_acc),))
        # Display the test accuracy using keras evaluation method.
        # _, keras_test_acc = model.evaluate(test_it)
        # print(f"Keras acc = {keras_test_acc}")

        # saving the model and the learning curves plot for it if there is an improvement in the results compared to the previous epoch
        # min_save_acc is a limit for starting saving the models because at the beginning the accuracies keep increasing, but they are still low,
        # so it's a waste of storage space to save all these model
        if val_acc > min_save_acc and val_acc > best_accuracy_dict[net_name]:
            best_accuracy_dict[net_name] = val_acc

            history = {"loss": train_losses, "val_loss": test_losses, "accuracy": train_accuracies,
                       "val_accuracy": test_accuracies}
            # learning curves
            summarize_diagnostics(history, val_acc, net_name, data_augmentation, train_conv_layers)

            # save model
            model.save(f"Models/{net_name}_{data_augmentation}_{train_conv_layers}_{'%.6f' % val_acc}.h5")

        val_acc_metric.reset_states()

        print("Time taken: %.2fs" % (time.time() - start_time))


# This function prints and saves a plots a confusion matrix for a saved model
# I also use it for printing a short statistics report
def print_confusion_matrix(batch_size=32):
    # loading the saved model
    model = load_model('../Final_Save/Single_Type/Models/VGG16_r_u_q_True_False_0.772727.h5')
    net_name = "VGG16 (% 77.2) [test set]"

    # the test generator without data augmentation to see the true/clear result
    test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    test_datagen.mean = [0.485, 0.456, 0.406]

    # prepare iterators for iterating through the images using the data generators
    test_it = test_datagen.flow_from_directory('dataset_splits/test/',
                                               class_mode='categorical', batch_size=batch_size, target_size=(224, 224),
                                               classes=classes_labels_dict, shuffle=False)

    # getting the images files names for getting there labels
    files_names = test_it.filenames

    # getting the model predictions for the test images
    predictions = model.predict_generator(test_it)
    # evaluating the model accuracy using the keras function
    _, acc = model.evaluate(test_it)

    actual_labels = [test_it.class_indices[file_name.split("/")[0]] for file_name in files_names]

    # shor report for statistics and accuracies calculations
    print(f"Keras Evaluate Acc: {acc}")
    print(f"Manual Acc: {accuracy_score(actual_labels, np.argmax(predictions, axis=1))}")
    print("****************************************************")
    print(f"Classes Indices: {test_it.class_indices}")
    print(f"Files Names: {files_names}")
    print(f"Actual Labels:    {actual_labels}")
    print(len(actual_labels))
    print(f"Predicted Labels: {np.argmax(predictions, axis=1)}")
    print("****************************************************")

    # creating and plotting the confusion matrix
    cm = confusion_matrix(actual_labels, np.argmax(predictions, axis=1))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(net_name)
    plt.show()


# Infinite models training for_each_net times for each net of in nets_names each time epochs_num epochs
epochs_num = 800
batch_size = 128
loops_counter = 0
for_each_net = 1
nets_names = ["VGG16"]  # ["ResNet50", "VGG16"]

while True:
    print(f"Main Loop Number --> {loops_counter}")
    loops_counter += 1

    for net_name in nets_names:
        for i in range(for_each_net):
            print(f"Loop {i + 1} For {net_name}:")
            run_trainings(net_name, epochs_num=epochs_num, batch_size=batch_size, data_augmentation=True,
                          train_conv_layers=False)
        print("\n********************************************************************************")
    print("----------------------------------------------------------------------------------")

# print_confusion_matrix()
