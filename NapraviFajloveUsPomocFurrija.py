import numpy as np
from tensorflow import keras
from keras import layers
import json
import numpy as np
from sklearn.model_selection import train_test_split

#OVAJ ARRAY MORA DA IMA SHAPE (50,X)
def UcitajIzTxtFajla(fajl):
    with open(fajl, 'r') as f:
        my_list = json.load(f)

    return np.array(my_list)

def VratiTreningPodatke_y():
    list = []
    for i in range(0,90):
        list.append([0,1])
    for i in range(0,90):
        list.append([1,0])
    return np.array(list)

def Vrati_Trening_Y():
    y = []
    for i in range(0,100):
        y.append([1,0,0,0])
    for i in range(0,100):
        y.append([0,1,0,0])
    for i in range(0, 100):
        y.append([0, 0, 1, 0])
    for i in range(0, 100):
        y.append([0, 0, 0, 1])


    return np.array(y)


def popraviinput(nizNizova,trazeniInput):
    for j in range(0,len(nizNizova)):
        if(len(nizNizova[j])!=trazeniInput):
            for i in range(len(nizNizova[j]),trazeniInput):
                nizNizova[j].append(0)
def Vrati_Trening_X():
    techno = UcitajIzTxtFajla("8000-16000-54/techno54.txt")
    #popraviinput(techno, 240)
    heavymetal = UcitajIzTxtFajla("8000-16000-54/heavymetal54.txt")
    #popraviinput(heavymetal, 240)
    regge = UcitajIzTxtFajla("8000-16000-54/regge54.txt")
    #popraviinput(regge, 240)
    klasicna = UcitajIzTxtFajla("8000-16000-54/klasicna54.txt")
    #popraviinput(klasicna, 240)
    X = klasicna.tolist() + techno.tolist() +  heavymetal.tolist() + regge.tolist()

    return np.array(X).astype('float32')

#PRVO UCITAVAMO TRAINDATU



#x_testing = Vrati_Testing_Podatke_x()
#y_testing = Vrati_testing_Podatke_y()


#print(x_trening.shape)
#print(y_trening.shape)


#print(x_testing.shape)
#print(y_testing.shape)

#NadjiNajboljeParametre(200,1500,100,3,90,5)

def VratiProsecanAccuracy(brojNeurona , brojEpoha):

    X = Vrati_Trening_X()
    y = Vrati_Trening_Y()
    from sklearn.model_selection import KFold
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    acc= 0
    accuracyNaTreningu = 0
    for train_idx, val_idx in kfold.split(X,y):
        # Get the training and validation data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = keras.Sequential()
        model.add(keras.layers.Dense(54, input_dim=54, activation='relu'))
        model.add(keras.layers.Dense(brojNeurona, activation='relu'))
        model.add(keras.layers.Dense(brojNeurona, activation='relu'))
        model.add(keras.layers.Dense(brojNeurona, activation='relu'))
        model.add(keras.layers.Dense(4, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Train and evaluate your model on this fold

        history = model.fit(X_train, y_train, epochs=brojEpoha, verbose=2)
        accuracyNaTreningu += history.history['accuracy'][-1]

        loss, accuracy = model.evaluate(X_val, y_val)
        acc+=accuracy

    return acc/10 ,accuracyNaTreningu/10


def NadjiNajboljeParametre():
    maxacc = 0
    maxi = -1
    maxj = -1
    for i in range(900, 1300, 40):
        print("progress: i:" +str(i))

        najboljiZaIteracije = 0
        for j in range(9, 14, 1):
            prosecanacc = VratiProsecanAccuracy(i, j)
            if(najboljiZaIteracije > prosecanacc):
                print("Prestaje da bude dobro")
                break
            najboljiZaIteracije = prosecanacc
            if (prosecanacc > maxacc):
                maxacc = prosecanacc
                maxi = i
                maxj = j
    print("maxi : " + str(maxi) + " maxj :" + str(maxj) + " acc  :" + str(maxacc))



#NadjiNajboljeParametre()
#for i in range(4,35,4):
list = []



print(111111111111111)
list.append(VratiProsecanAccuracy(200 , 80))

list.append(VratiProsecanAccuracy(200 , 100))

list.append(VratiProsecanAccuracy(200 , 120))

list.append(VratiProsecanAccuracy(200 , 150))
list.append(VratiProsecanAccuracy(200 , 180))



#list.append(VratiProsecanAccuracy(500 , 500))



print(list)

 #   accNaTestu , accNaTreningu = VratiProsecanAccuracy(1200 , i)
 #   print(str(i) + ": ")
#   print("Razlika Trening i test")
#    print((accNaTreningu-accNaTestu)*100)

#NadjiNajboljeParametre()