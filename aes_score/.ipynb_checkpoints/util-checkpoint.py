import keras
import tensorflow as tf
import numpy as np
from keras.preprocessing import sequence
import keras.backend as K
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
K.set_session(tf.Session(config=config))
model = load_model('aes_score/model/best_model.h5')
model.predict(np.zeros((1,350)))

import pickle as pk
with open('aes_score/model/efvocab.pkl', 'rb') as vocab_file:
    vocab = pk.load(vocab_file)

def read_sentence(line , vocab):
    data_x = []
    for word in line.lower().split(' '):
        if word =='<num>':num_hit+=1
        if word in vocab:
            data_x.append(vocab[word])
        else:
            data_x.append(vocab['<unk>'])
    return data_x


def convert_to_dataset_friendly_scores(scores_array , ranges=[0 , 5]):
    low, high = ranges[0] , ranges[1]
    scores_array = scores_array * (high - low) + low
    return scores_array


def predict_cerf(text):
    text = text.strip()
    test_sentence = read_sentence(text , vocab)
    test_sentence  = sequence.pad_sequences([test_sentence], maxlen=350)
    cerf_level = round(convert_to_dataset_friendly_scores(model.predict(test_sentence).squeeze()))
    if cerf_level==0.0:cerf_level='A1'
    if cerf_level==1.0:cerf_level='A2'
    if cerf_level==2.0:cerf_level='B1'
    if cerf_level==3.0:cerf_level='B2'
    if cerf_level==4.0:cerf_level='C1'
    if cerf_level==5.0:cerf_level='C2'
    return cerf_level

