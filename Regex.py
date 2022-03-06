import  re

def zadanie1A(tekst):
    return re.sub('\d','',tekst)


def zadanie1B(tekst):
    return re.sub('<|>|/','',tekst)

def zadanie1C(tekst):
    return re.sub('\W', ' ', tekst)

def zadanie2(tekst):
    return re.findall('#+[a-zA-Z]{1,}', tekst)

def zadanie3(tekst):
    return re.findall('[:;][^\s].', tekst)





