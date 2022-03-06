import Funkcja_Czyszczaca
import Tekst
import Stemming
import Stop_Words

print("Tekst po czyszczeniu:", Funkcja_Czyszczaca.funkcja(Tekst.tekst_czyszczenie))
print()
print("Stemming:", Stemming.stemming(Tekst.text_stemming))
print()
print("Stop Words", Stop_Words.stop_words(Tekst.tekst_stopWords))



