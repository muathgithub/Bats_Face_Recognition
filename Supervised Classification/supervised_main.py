import sys

from keras.layers import Input, Dropout
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.applications import VGG16, ResNet50
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import warnings

# These warnings ignore is for hiding the warnings from the console for clean observation
# like the warning of new versions of specific function in the matplotlib you can make it a comment to see these warnings
warnings.filterwarnings("ignore")

BATS_NUMBER = 11
# These variables for saving the models best accuracy during the training
best_accuracy = -1
best_accuracy_dict = {"VGG16": -1, "ResNet50": -1}


# defining the VGG16 pre_trained model with costume top (fully connected)
def define_model(net_name):
    model = None

    input_tensor = Input(shape=(224, 224, 3))

    if net_name == "VGG16":
        model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    elif net_name == "ResNet50":
        model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    for layer in model.layers:
        layer.trainable = False

    # add new classifier layers
    top_model = Flatten()(model.layers[-1].output)
    top_model = Dense(4096, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                      bias_regularizer=regularizers.L2(1e-4),
                      activity_regularizer=regularizers.L2(1e-5))(top_model)
    top_model = Dropout(0.2)(top_model)
    top_model = Dense(1072, activation='relu', kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                      bias_regularizer=regularizers.L2(1e-4),
                      activity_regularizer=regularizers.L2(1e-5))(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(BATS_NUMBER, activation='softmax')(top_model)
    # define new model
    model = Model(inputs=model.inputs, outputs=output_layer)

    # training optimizers
    SGD_opt = SGD(lr=0.0001, momentum=0.9)
    Adam_opt = Adam(learning_rate=0.0001)
    RMSprop_opt = RMSprop(learning_rate=0.0001)
    model.compile(optimizer=Adam_opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# plot diagnostic learning curves
def summarize_diagnostics(history, acc, net_name):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(history.history['loss'], color='blue', label='train')
    axs[0].plot(history.history['val_loss'], color='orange', label='test')
    axs[0].set_title('Cross Entropy Loss')

    axs[1].plot(history.history['accuracy'], color='blue', label='train')
    axs[1].plot(history.history['val_accuracy'], color='orange', label='test')
    axs[1].set_title('Classification Accuracy')
    #
    # for ax in axs.flat:
    #     ax.set(xlabel='epoch', ylabel='y-label')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # save plot to file
    # filename = sys.argv[0].split('/')[-1]
    plt.savefig(f"Plots/Losses/{net_name}_{'%.6f' % acc}_plot.png")
    plt.close()


# This function creates and trains the model epochs times
def run_test_harness(epochs_num, net_name):
    global best_accuracy, best_accuracy_dict
    # define model
    model = define_model(net_name)
    # create data generator
    train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                                       width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
                                       vertical_flip=True)
    test_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
    # Data augmentations for the training
    # , rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True, vertical_flip=True

    # specify imagenet mean values for centering
    train_datagen.mean = [0.485, 0.456, 0.406]
    test_datagen.mean = [0.485, 0.456, 0.406]

    # prepare iterator (data generators)
    train_it = train_datagen.flow_from_directory('dataset_splits/train/',
                                                 class_mode='categorical', batch_size=128, target_size=(224, 224),
                                                 shuffle=True)
    # Printing the classes labels numbers
    print(train_it.class_indices)

    test_it = test_datagen.flow_from_directory('dataset_splits/test/',
                                               class_mode='categorical', batch_size=128, target_size=(224, 224))
    # fit model
    history = model.fit(train_it, steps_per_epoch=len(train_it),
                        validation_data=test_it, validation_steps=len(test_it), epochs=epochs_num, verbose=1)
    # evaluate model
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    if acc > best_accuracy_dict[net_name]:
        print(f"Acc: {acc}")
        best_accuracy_dict[net_name] = acc

        # learning curves
        summarize_diagnostics(history, acc, net_name)

        # save model
        filename = sys.argv[0].split('/')[-1]
        model.save(f"Models/{net_name}_{'%.6f' % acc}.h5")


# Infinite models training 10 times for each net of ["VGG16", "ResNet50"] each time 750 epochs for
epochs_num = 750
loops_counter = 0
for_each_net = 10
nets_names = ["VGG16", "ResNet50"]
while True:
    print(f"Main Loop Number --> {loops_counter}")
    loops_counter += 1

    for net_name in nets_names:
        for i in range(for_each_net):
            print(f"Loop {i + 1} For {net_name}:")
            run_test_harness(epochs_num, net_name)
        print("\n********************************************************************************")
    print("----------------------------------------------------------------------------------")
