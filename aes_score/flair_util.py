from flair.models import SequenceTagger
tagger: SequenceTagger = SequenceTagger.load_from_file(model_file='aes_score/flair/best-model.pt')
from flair.models import SequenceTagger
from nltk import tokenize
from flair.data import Sentence



def sentence_rate(s_preds):
    score = len([tag for tag in s_preds if tag!='O']) / len(s_preds)
    return score

def corpus_dectect(corpus):
    tagger: SequenceTagger = SequenceTagger.load_from_file(model_file='aes_score/flair/best-model.pt')
    print('Model is OK !!!!!!')

    sen_arry = [sen for i in corpus.strip().split('\n') for sen in tokenize.sent_tokenize(i)]
    def sentence_ner(sen):
        sentence = Sentence(sen , use_tokenizer=True)
        tagger.predict(sentence)
        ner_tmp ,ner_words,ner_tags = [],[],[]
        label = [[token.text , token.get_tag('ner').value , token.get_tag('ner').score] for token in sentence]
        word_token , label_token , pro_token = [i[0] for i in label] , [i[1] for i in label] , [i[2] for i in label]
        return word_token , label_token , sentence_rate(label_token)
    sent_info = [sentence_ner(i) for i in sen_arry]
    return sent_info

def sentence_ner(sen):
    sentence = Sentence(sen , use_tokenizer=True)
    tagger.predict(sentence)
    ner_tmp ,ner_words,ner_tags = [],[],[]
    label = [[token.text , token.get_tag('ner').value , token.get_tag('ner').score] for token in sentence]
    word_token , label_token , pro_token = [i[0] for i in label] , [i[1] for i in label] , [i[2] for i in label]
    return word_token , label_token , sentence_rate(label_token)

def dectect_info(corpus):
    sen_info = corpus_dectect(corpus)
    info = { "sen_arry" :[],"tag_arry" :[] ,"score_arry" : []}
    for pair in sen_info:
        sen , tag , score = pair[0] , pair[1] , pair[2]
        #info["sen_arry"].append(' '.join(sen).strip().replace(' ,',',').replace(' .','.')+'\n')
        info["sen_arry"].append(sen)
        info["tag_arry"].append(tag)
        info["score_arry"].append(score)
    return info

def sen_dectect_info(sentence):
    tagger: SequenceTagger = SequenceTagger.load_from_file(model_file='aes_score/flair/best-model.pt')
    def sentence_ner(sen):
        sentence = Sentence(sen , use_tokenizer=True)
        tagger.predict(sentence)
        ner_tmp ,ner_words,ner_tags = [],[],[]
        label = [[token.text , token.get_tag('ner').value , token.get_tag('ner').score] for token in sentence]
        word_token , label_token , pro_token = [i[0] for i in label] , [i[1] for i in label] , [i[2] for i in label]
        return word_token , label_token , sentence_rate(label_token)
    sen , tag , score = sentence_ner(sentence)
    info = { "sen_arry" :[],"tag_arry" :[] ,"score_arry" : []}
    info["sen_arry"].append(sen)
    info["tag_arry"].append(tag)
    info["score_arry"].append(score)
    return info


print('Model is OK !!!!!!')
print(sentence_ner('You are a pig'))

