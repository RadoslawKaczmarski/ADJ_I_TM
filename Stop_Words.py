from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def stop_words(tekst:str) -> str:
    en_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tekst)
    filtred_text=[]

    for w in word_tokens:
        if w not in en_words:
            filtred_text.append(w)

    return filtred_text
