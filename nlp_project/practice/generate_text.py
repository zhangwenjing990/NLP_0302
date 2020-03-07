import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    text = raw_text.lower()
    return text

def text_to_chars(text):
    """完成字符去重与排序"""
    chars = list(set(text))
    sorted_chars = sorted(chars)
    return sorted_chars

def char_to_index(chars):
    """给每个字符一个索引"""
    char_dict = dict([(c, i) for i, c in enumerate(chars)])
    return char_dict

def index_to_char():
    index_dict = dict([(i, c) for i, c in enumerate(chars)])
    return index_dict

def generate_dataset(file_path):
    text = read_text(file_path)
    chars = text_to_chars(text)
    # todo



class CustomModel(keras.Model):
    def __init__(self, units, input_size, dropout_rate, output_size):
        super().__init__()
        self.lstm = LSTM(units, input_shape=input_size)
        self.drop = Dropout(dropout_rate)
        self.full = Dense(output_size, activation='softmax')

    def call(self, x):
        x = self.lstm(x)
        x = self.drop(x)
        output = self.full(x)
        return output

    # todo




if __name__ == '__main__':
    text = read_text('../data/Winston_Churchil.txt')
    chars = text_to_chars(text)
    char_to_index(chars)