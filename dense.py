import os
import random

import numpy as np
from keras.models import Sequential, Model
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
from keras.preprocessing import image
from keras.layers import Flatten, Dense, Softmax, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def generator_images_from_path(train_file_path, sample_dir, batch_count):
    train_file = open(train_file_path, 'r', errors='ignore')
    lines = train_file.readlines()

    while 1:
        random.shuffle(lines)
        train_images = []
        train_labels = []
        for i in range(batch_count):
            line = lines[i]
            image_id = line.split(" ")[0]
            image_id = image_id.split(".")[0]
            slabel = line.split(" ")[1]
            if slabel == "0":
                label = [0, 1]
            if slabel == "1":
                label = [1, 0]
            img_path = os.path.join(sample_dir, image_id, "ori.bmp")
            img = image.load_img(img_path, target_size=(224, 224))
            img_tensor = image.img_to_array(img)
            img_tensor = preprocess_input(img_tensor)
            train_images.append(img_tensor)
            train_labels.append(label)
        yield (np.array(train_images), np.array(train_labels))


convbase = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
x = convbase.output
x = GlobalAveragePooling2D()(convbase.output)
x = Dense(2, activation="softmax")(x)
model = Model(inputs=convbase.input, outputs=x)

model.compile(optimizer="rmsprop", loss=categorical_crossentropy, metrics=["acc"])
print(model.summary())

spath = "/home/liuml/keras/0/samples/"
trainfpath = "/home/liuml/maskrcnn/train.txt"
testfpath = "/home/liuml/maskrcnn/test.txt"
bc = 10

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

tensorboard = TensorBoard(log_dir="./logs",
                          histogram_freq=0, write_graph=True, write_images=False)
modelcheckpoint = ModelCheckpoint("./logs/weight.{epoch:02d}.h5",
                                  verbose=0, save_weights_only=True)
earlystop = EarlyStopping(monitor='acc', min_delta=0.001, patience=2, mode='max')

model.fit_generator(validation_steps=10,
                    generator=generator_images_from_path(train_file_path=trainfpath, sample_dir=spath,
                                                         batch_count=bc), epochs=10,
                    steps_per_epoch=100, callbacks=[tensorboard, modelcheckpoint, earlystop])
model.save_weights("densenet121.h5")
# model.load_weights("densenet121.h5")
result = model.evaluate_generator(generator=generator_images_from_path(train_file_path=testfpath, sample_dir=spath,
                                                                       batch_count=bc), steps=10)
print("accuracy: %s" % result[1])
