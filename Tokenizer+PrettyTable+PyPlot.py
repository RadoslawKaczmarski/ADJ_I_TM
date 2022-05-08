import numpy as np
import sklearn.feature_extraction.text
from PrzygotowanieDanych import tekstClear
import pandas as pd
import matplotlib.pyplot as plt
from prettytable import PrettyTable

df = pd.read_csv(r"Pliki_CSV/dataset.csv", nrows=10000)

# 1. Top 10 najczęściej występujących tokenów dla kolumny Text?
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=tekstClear)
X_transform = vectorizer.fit_transform(df['text'])
sum_list = X_transform.toarray().sum(axis=0)
suma = -np.sort(-sum_list)[:10]
slowa = vectorizer.get_feature_names_out()[-np.argsort(-sum_list)[:10]]

# Wykres
plt.subplots(figsize=(10,5))
plt.bar(slowa, suma, width=0.4, color='green')
plt.title("Top 10 najczęściej występujących tokenów dla kolumny Text")
plt.show()
print()

# PrettyTable
newTable = PrettyTable()
newTable.add_column("Słowa", slowa)
newTable.add_column("Suma", suma)
print(newTable)

################################################################################################################

# 2. Top 10 najważniejszych tokenów dla kolumny Text?
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=tekstClear)
X_transform = vectorizer.fit_transform(df['text'])
sum_list = X_transform.toarray().sum(axis=1)
sumaTFIDF = -np.sort(-sum_list).round(2)[:10]
slowaTFIDF = vectorizer.get_feature_names_out()[-np.argsort(-sum_list)[:10]]

#Wykres
plt.subplots(figsize=(11, 5))
plt.bar(slowaTFIDF, sumaTFIDF, width=0.5, color='red')
plt.title("Top 10 najważniejszych tokenów dla kolumny Text")
plt.show()
print()

# PrettyTable
newTable = PrettyTable()
newTable.add_column("Słowa TFIDF", slowaTFIDF)
newTable.add_column("Suma TFIDF", sumaTFIDF)
print(newTable)

#####################################################################################################################

# 3. Top 10 dokumentów, które zawierają najwięcej tokenów
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=tekstClear)
X_transform = vectorizer.fit_transform(df['text'])
z4 = X_transform.toarray().sum(axis=1)
copy_z4 = z4.copy()
liczbyTop10 = []
for i in range(10):
    index = np.argmax(copy_z4)
    liczbyTop10.append(index)
    copy_z4[index] = 0
slowaTop10 = vectorizer.get_feature_names_out()[liczbyTop10]

# Wykres
plt.subplots(figsize=(16, 5))
plt.bar(slowaTop10, liczbyTop10, width=0.5)
plt.title("Top 10 dokumentów, które zawierają najwięcej tokenów")
plt.show()
print()

# PrettyTable
newTable = PrettyTable()
newTable.add_column("Liczby Top 10", slowaTop10)
newTable.add_column("Słowa Top 10", liczbyTop10)
print(newTable)
