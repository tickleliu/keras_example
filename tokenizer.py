import keras.preprocessing.text as T
from keras.preprocessing.text import Tokenizer

text1='some thing to eat some thing to eat'
text2='some thing to drink some thing to drink'
texts=[text1,text2]

print(T.text_to_word_sequence(text1))  #以空格区分，中文也不例外 ['some', 'thing', 'to', 'eat']
# print(T.one_hot(text1,10))  #[7, 9, 3, 4] -- （10表示数字化向量为10以内的数字）
# print(T.one_hot(text2,10))  #[7, 9, 3, 1]

tokenizer = Tokenizer(num_words=3) #num_words:None或整数,处理的最大单词数量。少于此数的单词丢掉
tokenizer.fit_on_texts(texts)
print( tokenizer.word_counts) #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)]
print( tokenizer.word_index) #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
# print( tokenizer.word_docs) #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
# print( tokenizer.index_docs) #{1: 2, 2: 2, 3: 2, 4: 1, 5: 1}
#
# # num_words=多少会影响下面的结果，行数=num_words
from keras.preprocessing.sequence import pad_sequences
print( pad_sequences(tokenizer.texts_to_sequences(texts), 10)) #得到词索引[[1, 2, 3, 4], [1, 2, 3, 5]]
# print( tokenizer.texts_to_matrix(texts))  # 矩阵化=one_hot

