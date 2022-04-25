import numpy as np
import sklearn.feature_extraction.text
from przygotowanie_danych import tekst_tokenizer
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable


df = pd.read_csv(r"Pliki\Fake.csv")


# 1. Jeśli do vectorizera liczebnościowego przekażemy jedynie jeden dokument, to jakie
# wartości będzie miała otrzymana macierz? Albo jakich nie będzie miała?
# --Nie bedzie wartośc 0


# 2. Jak wyciągnąć top 10 najczęściej występujących tokenów?
# Sumować po kolumnach i posortowac tę sumę
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=tekst_tokenizer)
X_transform = vectorizer.fit_transform(df['title'][:5])
sum_list = X_transform.toarray().sum(axis=0)
suma = -np.sort(-sum_list)[:10]
slowa = vectorizer.get_feature_names_out()[-np.argsort(-sum_list)[:10]]

# print(f"Suma: {suma}")
# print(f"Słowa: {slowa}")
plt.subplots(figsize=(10,5))
plt.bar(slowa, suma, width=0.4, color='green')
plt.show()
print()

#PrettyTable
columns = ["Słowa","Suma"]
newTable = PrettyTable()
newTable.add_column(columns[0], slowa)
newTable.add_column(columns[1], suma)
print(newTable)


# 3. Jak wyciągnąć top 10 najważniejszych tokenów?
# --Wektorize tfidf i to samo co  w 2
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=tekst_tokenizer)
X_transform = vectorizer.fit_transform(df['title'][:10])
sum_list = X_transform.toarray().sum(axis=1)
sumaTFIDF = -np.sort(-sum_list).round(2)[:10]
slowaTFIDF = vectorizer.get_feature_names_out()[-np.argsort(-sum_list)[:10]]

# print(f"Suma TFIDF: {sumaTFIDF}")
# print(f"Słowa TFIDF : {slowaTFIDF}")
plt.subplots(figsize=(11, 5))
plt.bar(slowaTFIDF, sumaTFIDF, width=0.5, color='red')
plt.show()
print()

#PrettyTable
columns = ["Słowa TFIDF", "Suma TFIDF"]
newTable = PrettyTable()
newTable.add_column(columns[0], slowaTFIDF)
newTable.add_column(columns[1], sumaTFIDF)
print(newTable)



# 4. Jak wyciągnąć top 10 dokumentów, które zawierają najwięcej tokenów?
z4 = X_transform.toarray().sum(axis=1)
liczbaTop10 = (-z4).argsort()[:10]
slowaTop10 = vectorizer.get_feature_names_out()[(-z4).argsort()[:10]]

# print(f"Top 10 dokumentów: {(-z4).argsort()[:10]}")
# print(f"Top 10 dokumentów: {vectorizer.get_feature_names_out()[(-z4).argsort()[:10]]}")
plt.subplots(figsize=(11, 5))
plt.bar(slowaTop10, liczbaTop10, width=0.5)
plt.show()
print()

#PrettyTable
columns = ["Liczba Top 10", "Słowa Top 10"]
newTable = PrettyTable()
newTable.add_column(columns[0], slowaTop10)
newTable.add_column(columns[1], liczbaTop10)
print(newTable)
