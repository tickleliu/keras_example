import os

imdb_dir = os.path.join(os.getcwd(), "aclImdb")
train_dir = os.path.join(imdb_dir, "train")

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname)) as fopen:
                texts.append(fopen.read())
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100
training_samples = 200
validation_samples = 200
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences=sequences, maxlen=maxlen)

import numpy as np

labels = np.asarray(labels)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[: training_samples]
y_trian = labels[: training_samples]

x_test = data[training_samples: validation_samples + training_samples]
y_test = labels[training_samples: validation_samples + training_samples]

glove_dir = os.getcwd()
embeddings_index = {}
with open(os.path.join(glove_dir, "glove.6B.100d.txt")) as f:
    for line in f.readlines():
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=np.float32)
        embeddings_index[word] = coefs

embeddings_dim = 100
embeddings_matrix = np.zeros((max_words, embeddings_dim))
for word, i in word_index.items():
    embeddings_vector = embeddings_index.get(word)
    if i < max_words:
        if embeddings_vector is not None:
            embeddings_matrix[i] = embeddings_vector

from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embeddings_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()

model.layers[0].set_weights([embeddings_matrix])
model.layers[0].trainable = False

model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['acc'])
histroy = model.fit(x_train, y_trian, epochs=10, batch_size=32, validation_split=0.2)
model.save_weights("text.h5")

