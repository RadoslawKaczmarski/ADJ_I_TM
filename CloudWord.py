from Regex_Dane import tekst_clear
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv(r'Pliki_CSV/alexa_reviews.csv', nrows=1000, encoding='cp1252', sep=';')


def cloud(words: list) -> dict:
    bow = {}
    for w in words:
        if w not in bow.keys():
            bow[w] = 1
        else:
            bow[w] += 1
    return bow


# Cloud world dla kolumn verified_reviews
def cloud_word():
    tekst = ""
    for i in range(len(df['verified_reviews']))[:300]:
        tekst += df['verified_reviews'].iloc[i]
        text_wordcloud = tekst_clear(tekst)
        bow = cloud(text_wordcloud)

    wc = WordCloud()
    wc.generate_from_frequencies(bow)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("Cloud World for 'verified_reviews'")
    plt.show()
