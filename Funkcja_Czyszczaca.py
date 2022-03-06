import re


def funkcja(tekst:str) -> str:
    emotikony = re.findall('[:;][^\s].', tekst)
    tekst = tekst.lower()
    tekst = re.sub('\d','',tekst)
    tekst = re.sub('\W','',tekst)
    tekst = re.sub('[\s]{2}','',tekst)

    print("Lista emotikonek:", emotikony)
    for i in emotikony:
        tekst += str(i) + ' '
    return tekst
