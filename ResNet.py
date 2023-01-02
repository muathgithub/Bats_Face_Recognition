import sys
from keras.layers import Input
from matplotlib import pyplot
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator

best_accuracy = -1


# defining the ResNet50 pre_trained model with costume top (fully connected)
def define_model():
    input_tensor = Input(shape=(224, 224, 3))

    model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # for layer in model.layers:
    #     layer.trainable = False

    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.0001, momentum=0.9)
    adam_opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=adam_opt, loss='binary_crossentropy', metrics=['accuracy'])

    return model


# plot diagnostic learning curves
def summarize_diagnostics(history, acc):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(f"loss_plots/{filename[:-3]}_{'%.6f' % acc}_plot.png")
    pyplot.close()


# run the test harness for evaluating a model
def run_test_harness():
    global best_accuracy
    # define model
    model = define_model()
    # create data generator
    datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, rotation_range=20,
                                 width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True,
                                 vertical_flip=True, validation_split=0.2)
    # specify imagenet mean values for centering
    datagen.mean = [123.68, 116.779, 103.939]
    # prepare iterator
    train_it = datagen.flow_from_directory('dataset_gray_vs_niger_bats/train/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224), shuffle=True)

    # this part of the script plots the images that the train loader gets
    # in order to examine the data augmentation
    # print(train_it.class_indices)
    # for X_batch, y_batch in train_it:
    #     print(X_batch.min(), X_batch.mean(), X_batch.max())
    #     # create a grid of 3x3 images
    #     ran = 10
    #     fig, ax = pyplot.subplots(ran, ran, sharex=True, sharey=True, figsize=(ran, ran))
    #     for i in range(ran):
    #         for j in range(ran):
    #             ax[i][j].imshow(X_batch[i * 3 + j])
    #     # show the plot
    #     pyplot.show()

    test_it = datagen.flow_from_directory('dataset_gray_vs_niger_bats/test/',
                                          class_mode='binary', batch_size=64, target_size=(224, 224))
    # fit model
    history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                  validation_data=test_it, validation_steps=len(test_it), epochs=40, verbose=1)
    # evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))

    if acc > best_accuracy:
        best_accuracy = acc

        # learning curves
        summarize_diagnostics(history, acc)

        # save model
        filename = sys.argv[0].split('/')[-1]
        model.save(f"saved_models/{filename[:-3]}_{'%.6f' % acc}.h5")


loops_counter = 0
while True:
    print(f"Loop Number --> {loops_counter}")
    loops_counter += 1

    # entry point, run the test harness
    run_test_harness()

    print("----------------------------------------------------------------------------------")
