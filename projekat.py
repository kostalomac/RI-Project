
import numpy as np
from tensorflow import keras
from keras import layers
import json
import numpy as np
from sklearn.model_selection import train_test_split

# OVAJ ARRAY MORA DA IMA SHAPE (50,X)
def UcitajIzTxtFajla(fajl,velicinaTreningPlusValidacija):
    with open(fajl, 'r') as f:
        my_list = json.load(f)

    return np.array(my_list)[:velicinaTreningPlusValidacija]
def UcitajTestingPodatke(fajl):
    with open(fajl, 'r') as f:
        my_list = json.load(f)

    return np.array(my_list)[90:]
def VratiTestingY():
    y = []
    for i in range(0, 10):
        y.append([1, 0, 0, 0])
    for i in range(0, 10):
        y.append([0, 1, 0, 0])
    for i in range(0, 10):
        y.append([0, 0, 1, 0])
    for i in range(0, 10):
        y.append([0, 0, 0, 1])

    return np.array(y)

def Vrati_Trening_Y(velicinaTreningPlusValidacija):
    y = []
    for i in range(0, velicinaTreningPlusValidacija):
        y.append([1, 0, 0, 0])
    for i in range(0, velicinaTreningPlusValidacija):
        y.append([0, 1, 0, 0])
    for i in range(0, velicinaTreningPlusValidacija):
        y.append([0, 0, 1, 0])
    for i in range(0, velicinaTreningPlusValidacija):
        y.append([0, 0, 0, 1])

    return np.array(y)


def Vrati_Testing_X(Ime):
    ImfeFoldera = str(Ime)
    techno = UcitajTestingPodatke(ImfeFoldera + "/techno" + ImfeFoldera + ".txt")
    heavymetal = UcitajTestingPodatke(ImfeFoldera + "/heavymetal" + ImfeFoldera + ".txt")
    regge = UcitajTestingPodatke(ImfeFoldera + "/regge" + ImfeFoldera + ".txt")
    klasicna = UcitajTestingPodatke(ImfeFoldera + "/klasicna" + ImfeFoldera + ".txt")

    X = klasicna.tolist() + techno.tolist() + heavymetal.tolist() + regge.tolist()

    return np.array(X).astype('float32')


def Vrati_Trening_X(Ime,velicinaTreningPlusValidacija):
    ImfeFoldera = str(Ime)
    techno = UcitajIzTxtFajla(ImfeFoldera + "/techno" + ImfeFoldera + ".txt",velicinaTreningPlusValidacija)
    heavymetal = UcitajIzTxtFajla(ImfeFoldera + "/heavymetal" + ImfeFoldera + ".txt",velicinaTreningPlusValidacija)
    regge = UcitajIzTxtFajla(ImfeFoldera + "/regge" + ImfeFoldera + ".txt",velicinaTreningPlusValidacija)
    klasicna = UcitajIzTxtFajla(ImfeFoldera + "/klasicna" + ImfeFoldera + ".txt",velicinaTreningPlusValidacija)

    X = klasicna.tolist() + techno.tolist() + heavymetal.tolist() + regge.tolist()

    return np.array(X).astype('float32')


def VratiProsecanAccuracy(brojNeurona, brojEpoha, VelicinaUlaza,brojKfoldSplitova,greske,velicinaTreningPlusValidacija,zaDesetinu):
    PodaciOTacnost = [[0, 0] for _ in range(brojEpoha+1)]
    y = Vrati_Trening_Y(velicinaTreningPlusValidacija)
    X = Vrati_Trening_X(VelicinaUlaza, velicinaTreningPlusValidacija)
    if(zaDesetinu!=-1):
        velicinaJedneDesetine = int(VelicinaUlaza / 10)
        X = [sublist[velicinaJedneDesetine*zaDesetinu:velicinaJedneDesetine*(zaDesetinu+1)] for sublist in X]
        X = np.array(X)


    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=brojKfoldSplitova, shuffle=True, random_state=42)
    acc = 0
    model = 0
    indexDoKojeEpoheTrebaDaIde = brojEpoha
    for train_idx, val_idx in kfold.split(X, y):
        # Get the training and validation data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = keras.Sequential()
        if(zaDesetinu==-1):
            model.add(keras.layers.Dense(VelicinaUlaza, input_dim=VelicinaUlaza, activation='relu'))
        else:
            model.add(keras.layers.Dense(int(VelicinaUlaza/10), input_dim=int(VelicinaUlaza/10), activation='relu'))
        model.add(keras.layers.Dense(brojNeurona, activation='relu'))
        model.add(keras.layers.Dense(brojNeurona, activation='sigmoid'))
        model.add(keras.layers.Dense(brojNeurona, activation='relu'))
        model.add(keras.layers.Dense(brojNeurona, activation='sigmoid'))
        model.add(keras.layers.Dense(4, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Train and evaluate your model on this fold
        maximalnaVrednostNaTestu = 0
        kolikoPutaSmeDaPogresi = greske
        for i in range(0,indexDoKojeEpoheTrebaDaIde+1):
            history = model.fit(X_train, y_train, epochs=1, verbose=0)
            PodaciOTacnost[i][0] += history.history['accuracy'][-1]
            loss, accuracy = model.evaluate(X_val, y_val)
            PodaciOTacnost[i][1] += accuracy


            if(accuracy>=maximalnaVrednostNaTestu):
                maximalnaVrednostNaTestu = accuracy
                kolikoPutaSmeDaPogresi = greske
            else:
                if(history.history['accuracy'][-1]>accuracy):
                    kolikoPutaSmeDaPogresi -= 1
                indexDoKojeEpoheTrebaDaIde = i
                if(kolikoPutaSmeDaPogresi == 0):
                    break

    for i in range(0,len(PodaciOTacnost)):
        PodaciOTacnost[i][0] = PodaciOTacnost[i][0] / brojKfoldSplitova
        PodaciOTacnost[i][1] = PodaciOTacnost[i][1] / brojKfoldSplitova
    return indexDoKojeEpoheTrebaDaIde,PodaciOTacnost ,model


def VratiModelZaEksportovanje(brojEpoha,velicinaUlaza):
    model = keras.Sequential()
    model.add(keras.layers.Dense(velicinaUlaza, input_dim=velicinaUlaza, activation='relu'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(100, activation='sigmoid'))
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dense(100, activation='sigmoid'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    y = Vrati_Trening_Y(90)
    X = Vrati_Trening_X(velicinaUlaza, 90)
    model.fit(X, y,epochs=brojEpoha+1, verbose=0)
    return model



index ,podaci,model = VratiProsecanAccuracy(100, 100, 2400,5, 6,90,-1)
maximum = 0
for i in range(0,index+1):
    if(maximum<podaci[i][1]):
        maximum = podaci[i][1]


print(index)
print(podaci)
print(maximum)
modelzaEksport = VratiModelZaEksportovanje(index,2400)
loss, accuracy = modelzaEksport.evaluate(Vrati_Testing_X(2400),VratiTestingY())
modelzaEksport.save("mojModel.h5")
print(accuracy)
