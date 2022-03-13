from PrzygotowanieDanych import funkcja, stop_words, stemming, bag
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv(r"PlikiCSV\Fake.csv")

tekst = ""
for i in range(len(df['title'])):
    tekst += df['title'].iloc[i]

text_wordcloud = stemming(stop_words(funkcja(tekst)))
bow = bag(text_wordcloud)

wc = WordCloud()
wc.generate_from_frequencies(bow)
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()