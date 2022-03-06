from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


def stemming(txt:str) -> list:
    words = word_tokenize(txt)
    stemming_list=[]
    for word in words:
        stemming_list.append(ps.stem(word))
    return stemming_list
