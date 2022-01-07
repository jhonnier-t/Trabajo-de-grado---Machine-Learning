from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from joblib import dump
import cv2
import numpy as np
from imutils import paths
import os
import pickle
from matplotlib import pyplot as plt

modelo_i = 6000
Y = list()

def extract_color_stats1(image):
    R, G, B = cv2.split(image)
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
    np.std(G), np.std(B)]
    return features

def extract_color_stats2(image):
    (R, G, B), (H, S, V) = cv2.split(image),
    cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
    np.std(G), np.std(B),
    np.mean(H), np.mean(S), np.mean(V), np.std(H),
    np.std(S), np.std(V)]
    return features

def extract_color_stats3(image):
    H, S, V = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    features = [np.mean(H), np.mean(S), np.mean(V), np.std(H),
    np.std(S), np.std(V)]
    return features

models1 = {
    "knnRGB": KNeighborsClassifier(n_neighbors=1),
    "svmRGB": SVC(kernel='linear'),
    "logitRGB": LogisticRegression(solver='lbfgs',multi_class='auto'),
    "naive_bayesRGB" : GaussianNB(),
    "decision_treeRGB": DecisionTreeClassifier(),
    "random_forestRGB": RandomForestClassifier(n_estimators=100),
    "mlpBGR": MLPClassifier()
}

models2 = {
    "knnRGBHSV": KNeighborsClassifier(n_neighbors=1),
    "svmRGBHSV": SVC(kernel='linear'),
    "logitRGBHSV": LogisticRegression(solver='lbfgs',multi_class='auto'),
    "naive_bayesRGBHSV" : GaussianNB(),
    "decision_treeRGBHSV": DecisionTreeClassifier(),
    "random_forestRGBHSV":RandomForestClassifier(n_estimators=100),
    "mlpBGRHSV": MLPClassifier()
}

models3 = {
    "knnHSV": KNeighborsClassifier(n_neighbors=1),
    "svmHSV": SVC(kernel='linear'),
    "logitHSV": LogisticRegression(solver='lbfgs',multi_class='auto'),
    "naive_bayesHSV" : GaussianNB(),
    "decision_treeHSV": DecisionTreeClassifier(),
    "random_forestHSV": RandomForestClassifier(n_estimators=100),
    "mlpHSV": MLPClassifier()
}

pca = PCA(n_components=400, whiten=True)
svc = SVC(C=2, gamma=0.0001, kernel='linear')
knn = KNeighborsClassifier(n_neighbors=3)
logic = LogisticRegression(solver='lbfgs', multi_class='auto')
naive = GaussianNB()
dec_tree = DecisionTreeClassifier()
random_f = RandomForestClassifier(n_estimators=100)
mlp = MLPClassifier()

modelo = {
    "PCASVC": make_pipeline(pca, svc),
    "PCAKNN": make_pipeline(pca, knn),
    "PCALOGIC": make_pipeline(pca, logic),
    "PCANAIVE": make_pipeline(pca, naive),
    "PCATREE": make_pipeline(pca, dec_tree),
    "PCARANDOM": make_pipeline(pca, random_f),
    "PCAMLP": make_pipeline(pca, mlp)
}

pca = PCA(n_components=150, whiten=True)
svc = SVC(C=2, gamma=0.0001, kernel='linear')
modeloG = {
    "GPCASVC": make_pipeline(pca, svc),
    "GPCAKNN": make_pipeline(pca, knn),
    "GPCALOGIC": make_pipeline(pca, logic),
    "GPCANAIVE": make_pipeline(pca, naive),
    "GPCATREE": make_pipeline(pca, dec_tree),
    "GPCARANDOM": make_pipeline(pca, random_f),
    "GPCAMLP": make_pipeline(pca, mlp)
}

ruta = '/home/pi/Pruebas'
imagePaths = paths.list_images(ruta)
imagenes = []
imagenesG = []
caracteristicas1 = []
caracteristicas2 = []
caracteristicas3 = []
labels = []

for imagePath in imagePaths:
    image = cv2.cvtColor(cv2.imread(imagePath),cv2.COLOR_BGR2RGB)
    caracteristicas1.append(extract_color_stats1(image))
    caracteristicas2.append(extract_color_stats2(image))
    caracteristicas3.append(extract_color_stats3(image))
    imageG = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imagenes.append(np.array(image).flatten())
    imagenesG.append(np.array(imageG).flatten())
    labels.append(imagePath.split(os.path.sep)[-2])
    
print('Cargando Datos')
le = LabelEncoder()
labels = le.fit_transform(labels)
Titulos = ['Carro', 'Celda']
Xtrain1, Xtest1, Ytrain1, Ytest1 =
train_test_split(caracteristicas1, labels, test_size=0.2)
Xtrain2, Xtest2, Ytrain2, Ytest2 =
train_test_split(caracteristicas2, labels, test_size=0.2)
Xtrain3, Xtest3, Ytrain3, Ytest3 =
train_test_split(caracteristicas3, labels, test_size=0.2)
XtrainI, XtestI, YtrainI, YtestI = train_test_split(imagenes,
labels, test_size=0.2)
XtrainG, XtestG, YtrainG, YtestG = train_test_split(imagenesG,
labels, test_size=0.2)
XtrainI, XtestI, YtrainI, YtestI = np.array(XtrainI),
np.array(XtestI), np.array(YtrainI), np.array(YtestI)
XtrainG, XtestG, YtrainG, YtestG = np.array(XtrainG),
np.array(XtestG), np.array(YtrainG), np.array(YtestG)
rutM = '/home/pi/Modelos/'
np.set_printoptions(precision=4)


print('Entrenando primeros modelos')
for model, tipo in zip(models1.values(), models1.keys()):
    model.fit(Xtrain1, Ytrain1)
    accuracy = model.score(Xtest1, Ytest1)
    cm = confusion_matrix(Ytest1, model.predict(Xtest1),
    model.classes_,normalize='true')
    Y.append([tipo, cm])
    print('Modelo: {} --> Accuracy: {}'.format(tipo, accuracy))
    print(cm)
    dump(model, rutM+str(tipo)+str(modelo_i)+'.joblib')

print('Entrenando segundos modelos')

for model, tipo in zip(models2.values(), models2.keys()):
    model.fit(Xtrain2, Ytrain2)
    accuracy = model.score(Xtest2, Ytest2)
    cm = confusion_matrix(Ytest2, model.predict(Xtest2),
    model.classes_, normalize='true')
    Y.append([tipo, cm])
    print('Modelo: {} --> Accuracy: {}'.format(tipo, accuracy))
    print(cm)
    dump(model, rutM+str(tipo)+str(modelo_i)+'.joblib')

print('Entrenando Terceros modelos')

for model, tipo in zip(models3.values(), models3.keys()):
    model.fit(Xtrain3, Ytrain3)
    accuracy = model.score(Xtest3, Ytest3)
    cm = confusion_matrix(Ytest3, model.predict(Xtest3),
    model.classes_, normalize='true')
    Y.append([tipo, cm])
    print('Modelo: {} --> Accuracy: {}'.format(tipo, accuracy))
    print(cm)
    dump(model, rutM+str(tipo)+str(modelo_i)+'.joblib')

print('Entrenando ultimos modelos')
for model, tipo in zip(modelo.values(), modelo.keys()):
    model.fit(XtrainI, YtrainI)
    accuracy = model.score(XtestI, YtestI)
    cm = confusion_matrix(YtestI, model.predict(XtestI),
    model.classes_, normalize='true')
    Y.append([tipo, cm])
    print('Modelo: {} --> Accuracy: {}'.format(tipo, accuracy))
    print(cm)
    dump(model, rutM+str(tipo)+str(modelo_i)+'.joblib')

for model, tipo in zip(modeloG.values(), modeloG.keys()):
    model.fit(XtrainG, YtrainG)
    accuracy = model.score(XtestG, YtestG)
    cm = confusion_matrix(YtestG, model.predict(XtestG),
    model.classes_, normalize='true')
    Y.append([tipo, cm])
    print('Modelo: {} --> Accuracy: {}'.format(tipo, accuracy))
    print(cm)
    dump(model, rutM+str(tipo)+str(modelo_i)+'.joblib')

with open(rutM+'MatricesConfusion'+str(modelo_i)+'.pickle', 'wb') as F:
    pickle.dump(Y, F)
