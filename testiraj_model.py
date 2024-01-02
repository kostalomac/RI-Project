from tensorflow import keras
import numpy as np
import json
from sklearn.metrics import confusion_matrix
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

def Vrati_Testing_X(Ime):
    ImfeFoldera = str(Ime)
    techno = UcitajTestingPodatke(ImfeFoldera + "/techno" + ImfeFoldera + ".txt")
    heavymetal = UcitajTestingPodatke(ImfeFoldera + "/heavymetal" + ImfeFoldera + ".txt")
    regge = UcitajTestingPodatke(ImfeFoldera + "/regge" + ImfeFoldera + ".txt")
    klasicna = UcitajTestingPodatke(ImfeFoldera + "/klasicna" + ImfeFoldera + ".txt")

    X = klasicna.tolist() + techno.tolist() + heavymetal.tolist() + regge.tolist()

    return np.array(X).astype('float32')

model = keras.models.load_model("mojModel.h5")
loss, accuracy = model.evaluate(Vrati_Testing_X(2400),VratiTestingY())
pred = model.predict(Vrati_Testing_X(2400))

predicted_labels = np.argmax(pred, axis=1)
praveVrednosti = np.argmax(VratiTestingY(), axis=1)

conf_matrix = confusion_matrix(praveVrednosti, predicted_labels)
print(praveVrednosti)
print(predicted_labels)

print(conf_matrix)
print(accuracy)