# This script predict the given image class, and it saves the feature
# maps of specific convolutional layers according to the entered saved model
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.models import load_model
from keras.models import Model
from matplotlib import pyplot

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]

    return img


img_file_name = 'g_11.jpg'
# load the image
img = load_image(f'predection_images/{img_file_name}')
# load saved model
model = load_model('../saved_models/VGG_0.982143.h5')
# predict the class
result = model.predict(img)

print("Class: {} --> {}".format(result[0][0], str(round(result[0][0]))))
# print("Prob: {} %".format(1 - result[0][0]))

# the layers that we want to plot the features maps according to them
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i + 1].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)

feature_maps = model.predict(img)
# plot the output from each block
square = 8
plots_counter = 0

for layer_index, fmap in enumerate(feature_maps):

    # plot all 64 maps in an 8x8 squares
    ix = 1
    fig, ax = pyplot.subplots(square, square, figsize=(fmap.shape[1], fmap.shape[2]))
    for i in range(square):
        for j in range(square):
            # specify subplot and turn of axis

            ax[i][j].imshow(fmap[0, :, :, ix - 1])
            ix += 1

    # pyplot.title(model.layers[ixs[layer_index] + 1].name)
    # pyplot.savefig(f'feature_maps/{img_file_name[:-4]}_{plots_counter}.png')
    pyplot.savefig(f'feature_maps/layer_{ixs[layer_index] + 1}.png')
    plots_counter += 1
    # show the figure
    # pyplot.show()
