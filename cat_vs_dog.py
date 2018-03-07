import os
# import shutil
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#
#
# def prepare_data():
#     original_dataset_dir = os.path.join(os.getcwd(), "train")
#     base_dir = os.path.join(os.getcwd(), "train")
#     # os.mkdir(base_dir)
#
#     train_dir = os.path.join(base_dir, "train")
#     os.mkdir(train_dir)
#     val_dir = os.path.join(base_dir, "val")
#     os.mkdir(val_dir)
#     test_dir = os.path.join(base_dir, "test")
#     os.mkdir(test_dir)
#
#     train_dir_cat = os.path.join(train_dir, "cat")
#     os.mkdir(train_dir_cat)
#     val_dir_cat = os.path.join(val_dir, "cat")
#     os.mkdir(val_dir_cat)
#     test_dir_cat = os.path.join(test_dir, "cat")
#     os.mkdir(test_dir_cat)
#
#     train_dir_dog = os.path.join(train_dir, "dog")
#     os.mkdir(train_dir_dog)
#     val_dir_dog = os.path.join(val_dir, "dog")
#     os.mkdir(val_dir_dog)
#     test_dir_dog = os.path.join(test_dir, "dog")
#     os.mkdir(test_dir_dog)
#
#     fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
#     for fname in fnames:
#         src = os.path.join(original_dataset_dir, fname)
#         dst = os.path.join(train_dir_cat, fname)
#         shutil.copy(src, dst)
#
#     fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
#     for fname in fnames:
#         src = os.path.join(original_dataset_dir, fname)
#         dst = os.path.join(val_dir_cat, fname)
#         shutil.copy(src, dst)
#
#     fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
#     for fname in fnames:
#         src = os.path.join(original_dataset_dir, fname)
#         dst = os.path.join(test_dir_cat, fname)
#         shutil.copy(src, dst)
#
#     fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
#     for fname in fnames:
#         src = os.path.join(original_dataset_dir, fname)
#         dst = os.path.join(train_dir_dog, fname)
#         shutil.copy(src, dst)
#
#     fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
#     for fname in fnames:
#         src = os.path.join(original_dataset_dir, fname)
#     dst = os.path.join(val_dir_dog, fname)
#     shutil.copy(src, dst)
#
#     fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
#     for fname in fnames:
#         src = os.path.join(original_dataset_dir, fname)
#         dst = os.path.join(test_dir_dog, fname)
#         shutil.copy(src, dst)
#
#
# # prepare_data()
#
# from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
# from keras.models import Model
#
# # conv_base = VGG16(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
# # conv_base.trainable = False
# # model = Sequential()
# # model.add(conv_base)
# # model.add(Flatten())
# # model.add(Dense(256, activation='relu'))
# # model.add(Dense(2, activation='softmax'))
# input = Input(shape=(150, 150, 3))
# x = Conv2D(32, (3, 3), activation='relu')(input)
# x1 = x
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x2 = x
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x3 = x
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x4 = x
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = MaxPooling2D((2, 2))(x)
# x = Flatten()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(2, activation='softmax')(x)
# model = Model(inputs=[input], outputs=[x])
#
# from keras.optimizers import Adam
# from keras.losses import categorical_crossentropy
#
# model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['acc'])
# model.summary()
#
# from keras.preprocessing.image import ImageDataGenerator
#
# base_dir = os.path.join(os.getcwd(), "train")
# print(base_dir)
# train_dir = os.path.join(base_dir, "train")
# val_dir = os.path.join(base_dir, "val")
# train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
# test_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
#
# train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
#                                                     class_mode="categorical")
# test_generator = test_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=20,
#                                                   class_mode="categorical")
#
# for data_batch, label_batch in train_generator:
#     print("data batch shape", data_batch.shape)
#     print("label batch shape", label_batch.shape)
#     break
#
# # history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=test_generator, validation_steps=30)
# # model.save('cats_vs_dogs.h5')
#
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
#
# img = image.load_img(os.path.join(os.getcwd(), 'train', 'test', 'cat', 'cat.1849.jpg'), target_size=(150, 150))
# img_tensor = image.img_to_array(img)
# img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor /= 255.
#
# model = load_model("cats_vs_dogs.h5")
#
from keras.applications import VGG16
#
# model = VGG16(weights='imagenet', include_top=False)
# output_layers = [layer.output for layer in model.layers[: 8]]
# activation_model = Model(inputs=model.input, outputs=output_layers)
# activations = activation_model.predict(img_tensor)
# [print(activation.shape) for activation in activations]
#
# layer_name = "block4_conv1"
# # layer_name = "conv2d_2"
# filter_index = 30
#
from keras import backend as K
#
# layer_output = model.get_layer(layer_name).output
# loss = K.mean(layer_output[:, :, : filter_index])
# grads = K.gradients(loss, model.input)[0]
# print(grads.shape)
# grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
# iterator = K.function([model.input], [loss, grads])
#
import numpy as np
#
# loss_value, grads_value = iterator([np.zeros((1, 150, 150, 3))])
# input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
# for i in range(400):
#     loss_value, grads_value = iterator([input_img_data])
#     input_img_data += grads_value * 1.0
#
# def deprocess_image(x):
#      # normalize tensor: center on 0., ensure std is 0.1
#      x -= x.mean()
#      x /= (x.std() + 1e-5)
#      x *= 0.1
#      # clip to [0, 1]
#      x += 0.5
#      x = np.clip(x, 0, 1)
#      # convert to RGB array
#      x *= 255
#      x = np.clip(x, 0, 255).astype('uint8')
#      return x
# input_img_data = deprocess_image(input_img_data)


from keras.applications.vgg16 import preprocess_input, decode_predictions
model = VGG16(weights="imagenet")
img = image.load_img(os.path.join(os.getcwd(), 'train', 'cat.9991.jpg'), target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor = preprocess_input(img_tensor)
print(decode_predictions(model.predict(img_tensor)))
print(np.argmax(model.predict(img_tensor)[0]))
# This is the "african elephant" entry in the prediction vector
african_elephant_output = model.output[:, 285]
# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3')
# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))
# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
# These are the values of these two quantities, as Numpy arrays,
# given our sample image of two elephants
pooled_grads_value, conv_layer_output_value = iterate([img_tensor])
# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for i in range(512):
  conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)

# import matplotlib.pyplot as plt
# plt.imshow(input_img_data[0, :, :, :])
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
from skimage import io
io.imsave("heatmap.jpg", heatmap)
# plt.matshow(heatmap)
# plt.show()

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()
