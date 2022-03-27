import sklearn.feature_extraction.text
from PrzygotowanieDanych import tekst_tokenizer
import pandas as pd

df = pd.read_csv(r"Pliki\Fake.csv")


vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer = tekst_tokenizer)
X_transform = vectorizer.fit_transform(df['title'][:3])
print(vectorizer.get_feature_names_out())
print(X_transform.toarray())