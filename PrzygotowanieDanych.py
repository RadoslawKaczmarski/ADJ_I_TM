import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def funkcja(tekst: str) -> str:
    text_new = tekst

    # Usun Emotikony
    text_new = re.sub('[:;][^\s]', '', text_new)

    # Tekst male znaki
    text_new = text_new.lower()

    # Usun Cyfry
    text_new = re.sub('\d', ' ', text_new)

    # Usun znaki HTML
    text_new = re.sub(r"<[^>]*>", '', text_new)

    # Usun Whitespace
    text_new = re.sub('\s', ' ', text_new)

    # Usun znaki interpunkcyjne
    text_new = re.sub(r"[^0-9a-zA-Z ]+", '', text_new)

    return text_new


def stop_words(tekst: str) -> list:
    en_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tekst)
    return [w for w in word_tokens if w not in en_words]


def stemming(list_words: list) -> list:
    ps = PorterStemmer()
    return [ps.stem(w) for w in list_words]


def bag(words: list) -> dict:
    bow = {}
    for w in words:
        if w not in bow.keys():
            bow[w] = 1
        else:
            bow[w] += 1
    return bow