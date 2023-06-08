import sys
import warnings
from keras.models import Model
from keras.layers import Dense
from keras import regularizers
from keras.layers import Flatten
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.layers import Input, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16, ResNet50

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
min_save_acc = 0.45
best_accuracy_dict = {"VGG16": -1, "ResNet50": -1}

# choosing the needed labels dictionary
classes_labels_dict = bats_by_type_labels_dict
CLASSES_NUMBER = len(classes_labels_dict)


# defining the VGG16 pre_trained model with costume top (fully connected)
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

    # training optimizers
    Adam_opt = Adam(learning_rate=0.0001)
    # compiling the model with the training variables
    model.compile(optimizer=Adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# plot diagnostic learning curves (train and test losses and accuracies)
def summarize_diagnostics(history, acc, net_name):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(history.history['loss'], color='blue', label='train')
    axs[0].plot(history.history['val_loss'], color='orange', label='test')
    axs[0].set_title('Cross Entropy Loss')

    axs[1].plot(history.history['accuracy'], color='blue', label='train')
    axs[1].plot(history.history['val_accuracy'], color='orange', label='test')
    axs[1].set_title('Classification Accuracy')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # save plot to file
    plt.savefig(f"Plots/Losses/{net_name}_{'%.6f' % acc}_plot.png")
    plt.close()


# This function creates and trains the model epochs times
def run_trainings(net_name, epochs_num, batch_size, data_augmentation, train_conv_layers):
    # defining the model using the private function define_model()
    model = define_model(net_name, train_conv_layers=train_conv_layers)

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

    # Printing the classes labels numbers
    print(train_it.class_indices)

    test_it = test_datagen.flow_from_directory('dataset_splits/test/',
                                               class_mode='categorical', batch_size=128, target_size=(224, 224))
    # fit model using keras fit method
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it), epochs=epochs_num, verbose=1)

    # evaluate model using keras evaluate method
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    if acc > best_accuracy_dict[net_name] and acc > min_save_acc:
        print(f"Acc: {acc}")
        best_accuracy_dict[net_name] = acc

        # learning curves
        summarize_diagnostics(history, acc, net_name)

        # save model
        filename = sys.argv[0].split('/')[-1]
        model.save(f"Models/{net_name}_{'%.6f' % acc}.h5")


# Infinite models training for_each_net times for each net of in nets_names each time epochs_num epochs
epochs_num = 500
batch_size = 128
loops_counter = 0
for_each_net = 5
nets_names = ["ResNet50", "VGG16"]  # ["ResNet50", "VGG16"]

while True:
    print(f"Main Loop Number --> {loops_counter}")
    loops_counter += 1

    for net_name in nets_names:
        for i in range(for_each_net):
            print(f"Loop {i + 1} For {net_name}:")
            run_trainings(net_name, epochs_num=epochs_num, batch_size=batch_size, data_augmentation=True,
                          train_conv_layers=True)
        print("\n********************************************************************************")
    print("----------------------------------------------------------------------------------")
