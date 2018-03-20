import random
import numpy as np
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Input
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical

flatten = lambda l: [item for sublist in l for item in sublist]  # 二维展成一维
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]


def data_pipeline(data, length=50):
    data = [t[:-1] for t in data]  # 去掉'\n'
    # 数据的一行像这样：'BOS i want to fly from baltimore to dallas round trip EOS
    # \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # 分割成这样[原始句子的词，标注的序列，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]
    data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # 将BOS和EOS去掉，并去掉对应标注序列中相应的标注
    seq_in, seq_out, intent = list(zip(*data))
    sin = []
    sout = []
    # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
    for i in range(len(seq_in)):
        temp = seq_in[i]
        if len(temp) < length:
            temp.append('<EOS>')
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sin.append(temp)

        temp = seq_out[i]
        if len(temp) < length:
            while len(temp) < length:
                temp.append('<PAD>')
        else:
            temp = temp[:length]
            temp[-1] = '<EOS>'
        sout.append(temp)
        data = list(zip(sin, sout, intent))
    return data


def get_info_from_training_data(data):
    seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))
    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)
    # 生成word2index
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    # 生成index2word
    index2word = {v: k for k, v in word2index.items()}

    # 生成tag2index
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # 生成index2tag
    index2tag = {v: k for k, v in tag2index.items()}

    # 生成intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    # 生成index2intent
    index2intent = {v: k for k, v in intent2index.items()}
    return word2index, index2word, tag2index, index2tag, intent2index, index2intent


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch


def to_index(train, word2index, slot2index, intent2index):
    new_train = []
    for sin, sout, intent in train:
        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout_ix = list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"],
                           sout))
        intent_ix = intent2index[intent] if intent in intent2index else intent2index["<UNK>"]
        new_train.append([sin_ix, true_length, sout_ix, intent_ix])
    return new_train


input_steps = 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = 871
slot_size = 122
intent_size = 22
epoch_num = 50


def train(is_debug=False):
    # print(tf.trainable_variables())
    train_data = open("dataset/atis-2.train.w-intent.iob", "r").readlines()
    test_data = open("dataset/atis-2.dev.w-intent.iob", "r").readlines()
    train_data_ed = data_pipeline(train_data)
    test_data_ed = data_pipeline(test_data)
    word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
        get_info_from_training_data(train_data_ed)
    # print("slot2index: ", slot2index)
    # print("index2slot: ", index2slot)
    index_train = to_index(train_data_ed, word2index, slot2index, intent2index)
    index_test = to_index(test_data_ed, word2index, slot2index, intent2index)

    intents = [item[3] for item in index_train]
    intent_labels = np.eye(intent_size)[np.array(intents)]
    # intent_labels[:, np.array(intents)] = 1
    # intent_labels = to_categorical(index_train[2], num_classes=intent_size)
    intent_train = [item[0] for item in index_train]

    import tensorflow as tf
    from keras.layers import Lambda
    print_func = Lambda(lambda x: tf.Print(x, [tf.shape(x)]))
    input_voc = Input(shape=(input_steps,))
    embeding_voc = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=input_steps, mask_zero=True)(input_voc)
    lstm1 = Bidirectional(LSTM(units=hidden_size, dropout=0.7, return_sequences=True), merge_mode=None)(embeding_voc)
    print(len(lstm1))
    # lstm1 = print_func(lstm1)
    encoder = Bidirectional(LSTM(units=hidden_size, dropout=0.7, return_sequences=False))(lstm1)
    intent = Dense(intent_size, activation="sigmoid")(encoder)
    intent = Dense(intent_size, activation="softmax")(intent)
    model = Model(inputs=input_voc, outputs=intent)
    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["acc"])
    acc = model.fit(intent_train, intent_labels, batch_size=batch_size, epochs=1)
    model.save_weights("nlu.hdf5")
    model.load_weights("nlu.hdf5")

    intents = [item[3] for item in index_test]
    intent_labels = np.eye(intent_size)[np.array(intents)]
    intent_test = [item[0] for item in index_test]
    result = model.evaluate(intent_test, intent_labels)
    print(result)

if __name__ == "__main__":
    train()
