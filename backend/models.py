import numpy as np
import cv2
from joblib import load
from os import system, remove
from time import sleep
from matplotlib import pyplot as plt
from threading import Thread
from pickle import dump

cola = []
puestos = []
res, dim = (70, 170), (840, 450)
rutaF = '/home/pi/Plazas'
fichero = open(rutaF, 'wb')
fichero.close()
rutaM = '/home/pi/Modelos/PCARANDOM4000.joblib'
modelo = load(rutaM)
rutaI = '/home/pi/rutaI/Imagen.jpg'
cmdI = 'fswebcam -r 1920x1080 --jpeg 99 -D 3 -S 15 ' + rutaI
pts1 = np.float32([[900, 945], [1482, 259], [1102, 2], [495, 586]])
pts2 = np.float32([[0, 450], [840, 450], [840, 0], [0, 0]])
M = cv2.getPerspectiveTransform(pts1, pts2)
celdas = [[0, 74, 0, 170],[74, 137, 0, 170],
[137, 209, 0, 170],
[209, 280, 0, 170],
[280, 343, 0, 170],
[343, 416, 0, 170],
[416, 492, 0, 170],
[492, 555, 0, 170],
[555, 627, 0, 170],
[627, 701, 0, 170],
[701, 766, 0, 170],
[766, 840, 0, 170],
[0, 76, 280, 450],
[76, 138, 280, 450],
[138, 208, 280, 450],
[208, 279, 280, 450],
[279, 343, 280, 450],
[343, 414, 280, 450],
[414, 488, 280, 450],
[488, 551, 280, 450],
[551, 625, 280, 450]]

origenes = [(0, 170), (74, 170), (137, 170),
(209, 170), (280, 170), (343, 170),
(416, 170), (492, 170), (555, 170),
(627, 170), (701, 170), (766, 170),
(0, 280), (76, 280), (138, 280),
(208, 280), (279, 280), (343, 280),
(414, 280), (488, 280), (551, 280)]

cortar = lambda Img, X: cv2.resize(Img[X[2]: X[3], X[0]: X[1]],res)

def predict1(X):
    resultado = modelo.predict(X.flatten().reshape(1, -1))
    if resultado == 1:
        label = 'L'
    if resultado == 0:
        label= 'O'
    return label

def predict3(X):
    X = cv2.cvtColor(X, cv2.COLOR_RGB2GRAY)
    resultado = modelo.predict(X.flatten().reshape(1, -1))
    if resultado == 1:
        label = 'L'
    if resultado == 0:
        label= 'O'
    return label

def extract_color_stats(image):
    (R, G, B), (H, S, V) = cv2.split(image),
    cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
    np.std(G), np.std(B),
    np.mean(H), np.mean(S), np.mean(V), np.std(H),
    np.std(S), np.std(V)]
    return np.array(features).reshape(1, -1)

def predict2(X):
    X = extract_color_stats(X)
    resultado = modelo.predict(X)
    if resultado == 1:
        label = 'L'
    if resultado == 0:
        label = 'O'

    return label

#fig, axs = plt.subplots(3, 7)

def video(parametros):
    while True:
        if cola:
            frame = cola.pop(0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.destroyAllWindows()

t1 = Thread(name='Show', target=video, args=(1,))
t1.start()

while True:
    system(cmdI)
    imagen = cv2.cvtColor(cv2.imread(rutaI), cv2.COLOR_BGR2RGB)
    imagen1 = cv2.warpPerspective(imagen, M, dim)
    remove(rutaI)
    k, y = 0, 0
    puestos = list()
    try:
        fichero = open(rutaF, 'wb')
    except:
        pass
    for celda, corte in enumerate(celdas):
        #if (not celda%7 and celda !=0):
        #k -= 7
        #y += 1
        corta = cortar(imagen1, corte)
        #axs[y][k].imshow(corta)
        #axs[y][k].set_title(predict3(corta))
        puestos.append(predict1(corta))
        #k += 1
        try:
            dump(puestos, fichero)
            fichero.close()
        except:
            pass
        print(puestos)
        puestos = [puestos]
        cola.append(imagen1)
        #plt.show()
        sleep(5)