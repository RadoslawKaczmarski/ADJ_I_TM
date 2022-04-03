import numpy as np
import sklearn.feature_extraction.text
from przygotowanie_danych import tekst_tokenizer
import pandas as pd


df = pd.read_csv(r"Pliki\Fake.csv")


# vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer = tekst_tokenizer)
# X_transform = vectorizer.fit_transform(df['title'][:5])
# print(vectorizer.get_feature_names_out())
# print(X_transform.toarray())


# 1. Jeśli do vectorizera liczebnościowego przekażemy jedynie jeden dokument, to jakie
# wartości będzie miała otrzymana macierz? Albo jakich nie będzie miała?
# --Nie bedzie wartośc 0


# 2. Jak wyciągnąć top 10 najczęściej występujących tokenów?
# Sumować po kolumnach i posortowac tę sumę
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=tekst_tokenizer)
X_transform = vectorizer.fit_transform(df['title'][:5])
sum_list = X_transform.toarray().sum(axis=0)
print(f"Suma: {-np.sort(-sum_list)[:10]}")
print(f"Słowa: {vectorizer.get_feature_names_out()[-np.argsort(-sum_list)[:10]]}")


# 3. Jak wyciągnąć top 10 najważniejszych tokenów?
# --Wektorize tfidf i to samo co  w 2
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=tekst_tokenizer)
X_transform = vectorizer.fit_transform(df['title'][:10])
sum_list = X_transform.toarray().sum(axis=1)
print(f"Suma TFIDF: {-np.sort(-sum_list).round(2)[:10]}")
print(f"Słowa TFIDF : {vectorizer.get_feature_names_out()[-np.argsort(-sum_list)[:10]]}")


# 4. Jak wyciągnąć top 10 dokumentów, które zawierają najwięcej tokenów?
z4 = X_transform.toarray().sum(axis=1)
print(f"Top 10 dokumentów: {(-z4).argsort()[:10]}")
print(f"Top 10 dokumentów: {vectorizer.get_feature_names_out()[(-z4).argsort()[:10]]}")
