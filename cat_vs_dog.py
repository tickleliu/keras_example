import os
import shutil


def prepare_data():
    original_dataset_dir = os.path.join(os.getcwd(), "train")
    base_dir = os.path.join(os.getcwd(), "train")
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
        dst = os.path.join(train_dir_dog, fname)
        shutil.copy(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(val_dir_dog, fname)
    shutil.copy(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dir_dog, fname)
        shutil.copy(src, dst)

# prepare_data()

from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.applications import VGG16

input = Input(shape=(150, 150, 3))
conv_base = VGG16(include_top=False, input_shape=(150, 150, 3), weights='imagenet')
conv_base.trainable = False
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(2, activation='softmax'))
# x = Conv2D(32, (3, 3), activation='relu')(input)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(64, (3, 3), activation='relu')(x)
# x = BatchNormalization()(x)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = MaxPooling2D((2, 2))(x)
# x = Conv2D(128, (3, 3), activation='relu')(x)
# x = BatchNormalization()(x)
# x = Dropout(rate=0.5)(x)
# x = MaxPooling2D((2, 2))(x)
# x = Flatten()(x)
# x = Dropout(rate=0.5)(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(2, activation='softmax')(x)

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

base_dir = os.path.join(os.getcwd(), "train")
print(base_dir)
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
train_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode="categorical")
test_generator = test_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=20, class_mode="categorical")

for data_batch, label_batch in train_generator:
    print("data batch shape", data_batch.shape)
    print("label batch shape", label_batch.shape)
    break

history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=10, validation_data=test_generator, validation_steps=30)
model.save('cats_vs_dogs.h5')
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
