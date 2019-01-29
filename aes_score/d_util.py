import keras
import tensorflow as tf
import numpy as np
from nltk import tokenize
from keras.preprocessing import sequence
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "1"
K.set_session(tf.Session(config=config))
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.layers import concatenate, Input, LSTM, Dropout, Embedding , TimeDistributed ,Dense , Conv1D , MaxPooling1D , Flatten
from keras_contrib.layers import CRF
import numpy as np
from .preprocessing import get_vocab, read_input_files_text , pair_create , index_sents , get_char , ch_index_sents , load_file

tags = ['C','I','D','R']
tag2idx, idx2tag = get_vocab(tags, len(set(tags))+2)
embedding_matrix_flair = np.load('aes_score/prepare_data/fasttest-flair.npy')
word2idx, idx2word = load_file('aes_score/prepare_data/dectection.version2.vocab.pkl') , load_file('aes_score/prepare_data/dectection.version2.id.vocab.pkl')
MAX_VOCAB = len(word2idx)

txt_input = Input(shape=(None,), name='txt_input')
txt_embed = Embedding(MAX_VOCAB, 4396,name='txt_embedding',weights=[embedding_matrix_flair], mask_zero=True, trainable=False)(txt_input)
txt_drpot = Dropout(0.5, name='txt_dropout')(txt_embed)
lstm1 = Bidirectional(LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(txt_drpot)
lstm1 = Bidirectional(LSTM(256,dropout=0.2,recurrent_dropout=0.2, return_sequences=True))(lstm1)
crf = CRF(len(tags)+2, sparse_target=True)
mrg_chain = crf(lstm1)
model = Model(inputs=[txt_input], outputs=mrg_chain)

model.load_weights('aes_score/prepare_data/model.best.h5')
model.predict(np.zeros((1,4396)))


def sentence_input_index(sen_arry_token):
    text_x = index_sents(sen_arry_token, word2idx)
    X_test_sents = sequence.pad_sequences(text_x, maxlen=30, truncating='post', padding='post')
    return X_test_sents
def sentence_rate(s_preds):
    score = len([tag for tag in s_preds if tag!='C']) / len(s_preds)
    return score
def corpus_dectect(corpus):
    sen_arry = tokenize.sent_tokenize(corpus)
    sen_arry_token = [tokenize.word_tokenize(sen) for sen in sen_arry]
    X_test_sents = sentence_input_index(sen_arry_token)
    preds = model.predict([X_test_sents])
    preds = np.argmax(preds, axis=-1)
    s_preds_pair = [(sen_arry[index],[idx2tag[t] for t in s if idx2tag[t] != 'PAD']) for index , s in enumerate(preds)]
    sen_info = [[sen, sen_tag ,sentence_rate(sen_tag)] for sen , sen_tag in s_preds_pair]
    return sen_info

def custom_dectect_info(corpus):
    sen_info = corpus_dectect(corpus)
    info = { "sen_arry" :[],"tag_arry" :[] ,"score_arry" : []}
    for pair in sen_info:
        sen , tag , score = pair[0] , pair[1] , pair[2]
        info["sen_arry"].append(sen)
        info["tag_arry"].append(tag)
        info["score_arry"].append(score)
    return info