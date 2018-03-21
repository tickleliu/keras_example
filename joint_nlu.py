import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import numpy as np
import keras as K
from keras.layers import Dense, LSTM, Embedding, Bidirectional, Input
from keras.models import Model
from keras.losses import categorical_crossentropy

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
    vocab.add("<STR>")
    slot_tag = set(flatten(seq_out))
    slot_tag.add("<STR>")
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
        sin = ["<STR>"] + list(sin)
        sin_ix = list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"],
                          sin))
        true_length = sin.index("<EOS>")
        sout = ["<STR>"] + list(sout)
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
vocab_size = 872
slot_size = 123
intent_size = 22
epoch_num = 50


def train(is_debug=False):
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
    intent_train = [item[0] for item in index_train]
    intent_train = np.array(intent_train)
    slot_train = [item[2] for item in index_train]
    slot_train = np.array(slot_train)
    slot_train_target = np.insert(slot_train, slot_train.shape[1] - 1, values=0, axis=1)
    slot_train_target = np.delete(slot_train_target, 0, axis=1)
    slot_train_target = np.eye(slot_size)[slot_train_target]
    seq = np.array(index_test[0][0])

    import tensorflow as tf
    from keras.layers import Lambda
    print_func = Lambda(lambda x: tf.Print(x, [tf.shape(x)]))
    squeeze = Lambda(lambda x: tf.squeeze(x, axis=2))

    # encoder define
    input_voc = Input(shape=(None, 1))
    embedding_voc = Embedding(input_dim=vocab_size, output_dim=embedding_size,
                              mask_zero=True)
    embedding_voc_out = embedding_voc(input_voc)
    encoder_lstm1 = Bidirectional(LSTM(units=hidden_size, dropout=0.7, return_sequences=True), merge_mode="concat")
    embedding_voc_out = squeeze(embedding_voc_out)
    # embedding_voc_out = K.backend.squeeze(embedding_voc_out, axis=0)
    encoder_lstm1_out = encoder_lstm1(embedding_voc_out)

    # encoder
    encoder = Bidirectional(
        LSTM(units=hidden_size, dropout=0.7, return_sequences=False, return_state=True),
        merge_mode="concat")

    # encoder output, states
    encoder_out, forward_h, forward_c, backward_h, backward_c = encoder(encoder_lstm1_out)
    encoder_state = [forward_h, forward_c, backward_h, backward_c]

    # intent
    intent = Dense(intent_size, activation="linear")(encoder_out)
    intent = Dense(intent_size, activation="softmax")(intent)

    # encode state
    forward_h = K.layers.concatenate([forward_h, backward_h], 1)
    forward_c = K.layers.concatenate([forward_c, backward_c], 1)
    encoder_state = [forward_h, forward_c]

    # decoder define
    input_slot = Input(shape=(None, 1))
    embedding_slot = Embedding(input_dim=slot_size, output_dim=embedding_size,
                               mask_zero=True)
    embedding_slot_out = embedding_slot(input_slot)
    embedding_slot_out = squeeze(embedding_slot_out)
    decoder_lstm1 = LSTM(units=hidden_size * 2, dropout=0.7, return_state=True, return_sequences=True)
    decoder_lstm1_out, forward_h, forward_c = decoder_lstm1(
        embedding_slot_out, initial_state=[forward_h, forward_c])
    decoder = LSTM(units=hidden_size * 2, dropout=0.7, return_sequences=True, return_state=False)
    decoder_output = decoder(decoder_lstm1_out)
    dense1 = Dense(slot_size, activation="linear")
    dense1_out = dense1(decoder_output)
    dense2 = Dense(slot_size, activation="softmax")
    dense2_out = dense2(dense1_out)
    model = Model(inputs=[input_voc, input_slot], outputs=[intent, dense2_out])
    # print(model.summary())

    def intent_slot_loss(y_true, y_pred):
        y_slot_true = y_true[0]
        y_intent_true = y_true[1]
        y_slot_pred = y_pred[0]
        y_intent_pred = y_pred[1]

        return K.losses.categorical_crossentropy(y_slot_true, y_slot_pred) + K.losses.categorical_crossentropy(
            y_intent_true, y_intent_pred)

    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["acc"])
    # acc = model.fit([np.expand_dims(intent_train, 2), np.expand_dims(slot_train, 2)], [intent_labels, slot_train_target], batch_size=batch_size, epochs=1)
    # print(acc)
    # model.save_weights("nlu.hdf5")
    model.load_weights("nlu.hdf5")

    ## inference
    encoder = Model(input_voc, encoder_state)
    intenter = Model(input_voc, intent)
    # print(encoder.summary())

    # decoder
    decoder_in = Input(shape=(1,))
    decoder_state_in_h = Input(shape=(hidden_size * 2,))
    decoder_state_in_c = Input(shape=(hidden_size * 2,))
    decoder_state_in = [decoder_state_in_h, decoder_state_in_c]
    decoder_out, decoder_h, decoder_c = decoder_lstm1(embedding_slot(decoder_in), initial_state=decoder_state_in)
    decoder_state_out = [decoder_h, decoder_c] #output
    decoder_output = decoder(decoder_out)
    dense1_out = dense1(decoder_output)
    dense2_out = dense2(dense1_out) #output
    decoder_model = Model([decoder_in] + decoder_state_in, [dense2_out] + decoder_state_out)
    print(decoder_model.summary())

    i = 0
    print(seq)
    seq = np.expand_dims(np.array(seq), 0)
    seq = np.expand_dims(np.array(seq), 2)
    encoder_state = encoder.predict(seq)
    intent = intenter.predict(seq)
    intent = np.argmax(intent)
    print(index2intent[intent])

    index = slot2index["<STR>"]
    while i < input_steps:
        i = i + 1
        decoder_out, state_h, state_c = decoder_model.predict([np.array([index])] + encoder_state)
        index = np.argmax(decoder_out)
        encoder_state = [state_h, state_c]
        print(index2slot[index])
        # intents = [item[3] for item in index_test]
        # intent_labels = np.eye(intent_size)[np.array(intents)]
        # intent_test = [item[0] for item in index_test]
        #     # result = model.evaluate(np.array(intent_test), intent_labels)
        #     # print(result)


if __name__ == "__main__":
    train()
