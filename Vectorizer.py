import pandas as pd
import numpy as np
from przygotowanie_danych import tekst_tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import LinearSVC
from sklearn import metrics

df1 = pd.read_csv(r"Pliki/Fake.csv")
df2 = pd.read_csv(r"Pliki/True.csv")

df1["Fake/True"] = 1
df2["Fake/True"] = 0

df_main = pd.concat([df1, df2])
df_main.reset_index()

vectorizer = CountVectorizer(tokenizer=tekst_tokenizer)
df_vectorizer = vectorizer.fit_transform(df_main['title'])

X_train, X_test, Y_train, Y_test = train_test_split(df_vectorizer, df_main["Fake/True"], test_size=0.4, random_state=44)

klasyfiaktory = [DecisionTreeClassifier(),
                 RandomForestClassifier(),
                 LinearSVC(),
                 AdaBoostClassifier(),
                 BaggingClassifier()]

for klasyfikator in klasyfiaktory:
    klasyfikator.fit(X_train, Y_train)
    Y_pred = klasyfikator.predict(X_test)
    print(f"{klasyfikator} - Accuracy: {round(metrics.accuracy_score(Y_test,Y_pred)*100,2)}%")