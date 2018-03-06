import numpy as np
from gensim.models import Word2Vec
w2cmodel = Word2Vec.load("./50features_1minwords_10context")
POS_DIC = {"X":0,"O":1,"B":2,"I":3}
DIC_LEN = len(POS_DIC)


def get_basic_data(sentences_X,sentences_Y,sentences_len,filepath):
    f = open(filepath)
    list_X = []
    list_Y = []
    list_esc = []
    for line in f.readlines():
        if(line.strip()=="<end_for_sentence>"):
            sentences_X.append(list_X)
            sentences_len.append(len(list_X))
            sentences_Y.append(list_Y)
            list_X = []
            list_Y = []
        else:
            list_X.append(line.split(" ")[0].strip().lower())
            list_Y.append(line.split(" ")[1].strip())
            if line.split(" ")[1].strip() == 'B':
                list_esc.append(line.split(" ")[2].strip())
    return list_esc


if __name__ == "__main__":
    sentences_train_X = []
    sentences_test_X = []
    sentences_train_len = []
    sentences_train_Y = []
    sentences_test_Y = []
    sentences_test_len = []
    xx = get_basic_data(sentences_test_X,sentences_test_Y,sentences_test_len,'./data/test')
    get_basic_data(sentences_train_X,sentences_train_Y,sentences_train_len,'./data/train')
    print(  sentences_test_X  )
    print(  sentences_test_len )