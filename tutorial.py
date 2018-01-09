
import keras.preprocessing.text as T


text = '''
本文介绍keras提供的预处理包keras.preproceing下的text与序列处理模块sequence模块 2 text模块提供的方法 text_to_word_sequence(text,fileter) 可以
'''
print(T.text_to_word_sequence(text=text, filters=))
