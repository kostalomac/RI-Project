import json
import numpy as np


def UcitajIzTxtFajla(fajl):
    with open(fajl, 'r') as f:
        my_list = json.load(f)

    return np.array(my_list)

def Diskretizuj(ListaPesama,KolikoPutaSmanjiti):
    if(len(ListaPesama[0])%KolikoPutaSmanjiti!=0):
        print("Nije Deljivo")
        return
    NovaListaPesama = [0]*100
    for j in range(0,len(NovaListaPesama)):
        #print(j)
        PesmaKojaSeObradjuje = ListaPesama[j]
        duzinaNovogFajla = int(len(PesmaKojaSeObradjuje)/KolikoPutaSmanjiti)
        novFajl = [0]*duzinaNovogFajla
        for i in range(0,duzinaNovogFajla):
            novFajl[i] = sum(PesmaKojaSeObradjuje[i*KolikoPutaSmanjiti:i*KolikoPutaSmanjiti+KolikoPutaSmanjiti])/KolikoPutaSmanjiti
        NovaListaPesama[j] = novFajl
    return NovaListaPesama

def SacuvajUJson(imeFajla, data):
    # Save the list to a JSON file
    with open(imeFajla, 'w') as json_file:
        json.dump(data, json_file)


def NparaviFajlove(kolikoSmanjiti):
    Smanjenje = kolikoSmanjiti
    rege = UcitajIzTxtFajla("regge2400.txt")
    noveregePesme = Diskretizuj(rege, Smanjenje)
    SacuvajUJson("regge" + str(len(noveregePesme[0])) + ".txt", noveregePesme)

    klasicne = UcitajIzTxtFajla("klasicna2400.txt")
    noveklasicnePesme = Diskretizuj(klasicne, Smanjenje)
    SacuvajUJson("klasicna" + str(len(noveklasicnePesme[0])) + ".txt", noveklasicnePesme)

    techno = UcitajIzTxtFajla("techno2400.txt")
    novetechnoePesme = Diskretizuj(techno, Smanjenje)
    SacuvajUJson("techno" + str(len(novetechnoePesme[0])) + ".txt", novetechnoePesme)

    heavymetal = UcitajIzTxtFajla("heavymetal2400.txt")
    noveheavymetalPesme = Diskretizuj(heavymetal, Smanjenje)
    SacuvajUJson("heavymetal" + str(len(noveheavymetalPesme[0])) + ".txt", noveheavymetalPesme)


Smanjenje = 120
NparaviFajlove(Smanjenje)
#print(str(novePesme[0][0]) + " = " + str((rege[0][0]+rege[0][1])/2))

#print(len(novePesme[0]))

#print(a[0:2])


