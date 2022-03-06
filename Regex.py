import re


def zadanie_1a(tekst):
    return re.sub('\d', '', tekst)


def zadanie_1b(tekst):
    return re.sub('<|>|/', '', tekst)


def zadanie_1c(tekst):
    return re.sub('\W', ' ', tekst)


def zadanie2(tekst):
    return re.findall('#+[a-zA-Z]+', tekst)


def zadanie3(tekst):
    return re.findall('[:;][^\s].', tekst)
