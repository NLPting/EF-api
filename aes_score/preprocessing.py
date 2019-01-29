import codecs, re, random
from collections import Counter
import numpy as np

# network hyperparameters
MAX_LENGTH = 30

import pickle
def load_file(path):
    f = open(path,"rb")
    data = pickle.load(f)
    f.close()
    return data

def read_input_files_text(file_paths, max_sentence_length=-1):
    sentences = []
    line_length = None
    with open(file_paths, "r") as f:
        sentence = []
        for index , line in  enumerate(f):
            line = line.strip()
            try:
                if len(line) > 0:
                    line_parts = line.split('\t')
                    #print(line_parts)
                    assert(len(line_parts) >= 2)
                    assert(len(line_parts) == line_length or line_length == None)
                    line_length = len(line_parts)
                    sentence.append(line_parts)
                elif len(line) == 0 and len(sentence) > 0:
                    if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                        sentences.append(sentence)            
                    sentence = []
            except:
                print(index)
        if len(sentence) > 0:
            if max_sentence_length <= 0 or len(sentence) <= max_sentence_length:
                sentences.append(sentence)
                
    return sentences
def pair_create(data):
    x , y = [] , []
    for line in data[:]:
        s , t = [] , []
        for pair in line:
            s.append(pair[0])
            t.append(pair[1])
        x.append(s)
        y.append(t)
    return x , y

# function to get vocab, maxvocab
# takes sents : list (tokenized lists of sentences)
# takes maxvocab : int (maximum vocab size incl. UNK, PAD
# takes stoplist : list (words to ignore)
# returns vocab_dict (word to index), inv_vocab_dict (index to word)
def get_vocab(sent_toks, maxvocab=10000, min_count=1, stoplist=[], unk='UNK', pad='PAD', verbose=False):
    # get vocab list
    vocab = [word for sent in sent_toks for word in sent]
    sorted_vocab = sorted(Counter(vocab).most_common(), key=lambda x: x[1], reverse=True)
    sorted_vocab = [i for i in sorted_vocab if i[0] not in stoplist and i[0] != unk]
    if verbose:
        print("total vocab:", len(sorted_vocab))
    sorted_vocab = [i for i in sorted_vocab if i[1] >= min_count]
    if verbose:
        print("vocab over min_count:", len(sorted_vocab))
    # reserve for PAD and UNK
    sorted_vocab = [i[0] for i in sorted_vocab[:maxvocab - 2]]
    vocab_dict = {k: v + 1 for v, k in enumerate(sorted_vocab)}
    vocab_dict[unk] = len(sorted_vocab) + 1
    vocab_dict[pad] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}

    return vocab_dict, inv_vocab_dict


# function to convert sents to indexed vectors
# takes list : sents (tokenized sentences)
# takes dict : vocab (word to idx mapping)
# returns list of lists of indexed sentences
def index_sents(sent_tokens, vocab_dict, reverse=False, unk_name='UNK', verbose=False):
    vectors = []
    for sent in sent_tokens:
        sent_vect = []
        if reverse:
            sent = sent[::-1]
        for word in sent:
            if word in vocab_dict.keys():
                sent_vect.append(vocab_dict[word])
            else:  # out of max_vocab range or OOV
                sent_vect.append(vocab_dict[unk_name])
        vectors.append(np.asarray(sent_vect))
    vectors = np.asarray(vectors)
    return vectors

def get_char(sent_toks, min_count=1, stoplist=[], cunk='CUNK', pad='PAD', verbose=False):
    char_counter = Counter()
    for sentence in sent_toks:
        for word in sentence:
            char_counter.update(word[0])
    char2id = {}
    for char, count in char_counter.most_common():
        if char not in char2id:
            char2id[char] = len(char2id)
    vocab_dict = {k: v + 1 for v, k in enumerate(char2id)}
    vocab_dict[cunk] = len(char2id) + 1
    vocab_dict[pad] = 0
    inv_vocab_dict = {v: k for k, v in vocab_dict.items()}
    return vocab_dict, inv_vocab_dict

def ch_index_sents(sent_tokens, vocab_dict, reverse=False, unk_name='CUNK', verbose=False):
    vectors = []
    for sent in sent_tokens:
        sent_seq = []
        for i in range(30):
            word_seq = []
            for j in range(len(vocab_dict)):
                try:
                    word_seq.append(vocab_dict.get(sent[i][j][0])) 
                except:
                    word_seq.append(vocab_dict.get("PAD"))
            sent_seq.append(word_seq)
        vectors.append(sent_seq)
    return np.asarray(vectors)

# decode an integer-indexed sequence
# takes indexed_list : one integer-indexedf sentence (list or array)
# takes inv_vocab_dict : dict (index to word)
# returns list of string tokens
def decode_sequence(indexed_list, inv_vocab_dict):
    str = []
    for idx in indexed_list:
        # print(intr)
        str.append(inv_vocab_dict[int(idx)])
    return(str)