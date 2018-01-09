import os
import shutil


def prepare_data():
    original_dataset_dir = "/home/liuml/model_data/train"
    base_dir = "/home/liuml/model_data/train/"
    # os.mkdir(base_dir)

    train_dir = os.path.join(base_dir, "train")
    os.mkdir(train_dir)
    val_dir = os.path.join(base_dir, "val")
    os.mkdir(val_dir)
    test_dir = os.path.join(base_dir, "test")
    os.mkdir(test_dir)

    train_dir_cat = os.path.join(train_dir, "cat")
    os.mkdir(train_dir_cat)
    val_dir_cat = os.path.join(val_dir, "cat")
    os.mkdir(val_dir_cat)
    test_dir_cat = os.path.join(test_dir, "cat")
    os.mkdir(test_dir_cat)

    train_dir_dog = os.path.join(train_dir, "dog")
    os.mkdir(train_dir_dog)
    val_dir_dog = os.path.join(val_dir, "dog")
    os.mkdir(val_dir_dog)
    test_dir_dog = os.path.join(test_dir, "dog")
    os.mkdir(test_dir_dog)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dir_cat, fname)
        shutil.copy(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(val_dir_cat, fname)
        shutil.copy(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dir_cat, fname)
        shutil.copy(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dir_cat, fname)
        shutil.copy(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(val_dir_cat, fname)
    shutil.copy(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dir_cat, fname)
        shutil.copy(src, dst)


from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model

input = Input(shape=(150, 150, 3))
x = Conv2D(32, (3, 3), activation='relu')(input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(2, activation='softmax')(x)

model = Model(inputs=[input], outputs=[x])

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
