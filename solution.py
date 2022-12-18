from sklearn import tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier # librairie pour la forêt aléatoire
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from cleaner import WindowFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# La classe cleaner "nettoie" le fichier et renvoie soit les fenêtres de contexte selon la taille demandée.
cleaner = WindowFeatures()
vec = DictVectorizer()

# Préparation des données
# Fenêtre de contexte de la phrase au complet
pos_sentences, y_sentences = cleaner.GetFeaturesSentences('interest.acl94.txt')
pos_vectorized_sentences = vec.fit_transform(pos_sentences)
X_sentences = pos_vectorized_sentences
X_sentences_train, X_sentences_test, y_sentences_train, y_sentences_test =  train_test_split(X_sentences, y_sentences, test_size=0.33, random_state=42)

# Fenêtre de contexte de taille 1
pos_window1, y1 = cleaner.GetFeaturesWindow('interest.acl94.txt', 1)
pos_vectorized1 = vec.fit_transform(pos_window1)
X1 = pos_vectorized1
X1_train, X1_test, y1_train, y1_test =  train_test_split(X1, y1, test_size=0.33, random_state=42)

# Fenêtre de contexte de taille 2
pos_window2, y2 = cleaner.GetFeaturesWindow('interest.acl94.txt', 2)
pos_vectorized2 = vec.fit_transform(pos_window2)
X2 = pos_vectorized2
X2_train, X2_test, y2_train, y2_test =  train_test_split(X2, y2, test_size=0.33, random_state=42)

# Fenêtre de contexte de taille 3
pos_window3, y3 = cleaner.GetFeaturesWindow('interest.acl94.txt', 3)
pos_vectorized3 = vec.fit_transform(pos_window3)
X3 = pos_vectorized3
X3_train, X3_test, y3_train, y3_test =  train_test_split(X3, y3, test_size=0.33, random_state=42)

# Fenêtre de contexte de taille 4
pos_window4, y4 = cleaner.GetFeaturesWindow('interest.acl94.txt', 4)
pos_vectorized4 = vec.fit_transform(pos_window4)
X4 = pos_vectorized4
X4_train, X4_test, y4_train, y4_test =  train_test_split(X4, y4, test_size=0.33, random_state=42)

# Random Forest avec des fenêtres de contexte de tailles multiples
clf_rf = RandomForestClassifier(n_estimators=50)
clf_rf.fit(X1_train, y1_train)
y_pred = clf_rf.predict(X1_test) # prédictions à partir du modèle sur les données test
print('The accuracy of Random Forest with a features window of size 1 is :' + str(accuracy_score(y1_test, y_pred)))

clf_rf = RandomForestClassifier(n_estimators=50)
clf_rf.fit(X2_train, y2_train)
y_pred = clf_rf.predict(X2_test) # prédictions à partir du modèle sur les données test
print('The accuracy of Random Forest with a features window of size 2 is :' + str(accuracy_score(y2_test, y_pred)))

clf_rf = RandomForestClassifier(n_estimators=50)
clf_rf.fit(X3_train, y3_train)
y_pred = clf_rf.predict(X3_test) # prédictions à partir du modèle sur les données test
print('The accuracy of Random Forest with a features window of size 3 is :' + str(accuracy_score(y3_test, y_pred)))

clf_rf = RandomForestClassifier(n_estimators=50)
clf_rf.fit(X4_train, y4_train)
y_pred = clf_rf.predict(X4_test) # prédictions à partir du modèle sur les données test
print('The accuracy of Random Forest with a features window of size 4 is :' + str(accuracy_score(y4_test, y_pred)))

clf_rf = RandomForestClassifier(n_estimators=50)
clf_rf.fit(X_sentences_train, y_sentences_train)
y_pred = clf_rf.predict(X_sentences_test) # prédictions à partir du modèle sur les données test
print('The accuracy of Random Forest when using the whole sentence is :' + str(accuracy_score(y_sentences_test, y_pred)))

# Naive Bayes
clf_mnb = MultinomialNB()
clf_mnb.fit(X1_train, y1_train)
y_pred = clf_mnb.predict(X1_test) # prédictions à partir du modèle sur les données test
print('The accuracy of Naive Bayes when using the X2 is :' + str(accuracy_score(y1_test, y_pred)))

# SVM
clf_svm = svm.SVC()
clf_svm.fit(X1_train, y1_train)
y_pred = clf_svm.predict(X1_test) # prédictions à partir du modèle sur les données test
print('The accuracy of SVM when using X1 is :' + str(accuracy_score(y1_test, y_pred)))

# Scaling the data to avoid errors
scaler = StandardScaler(with_mean=False)
scaler.fit(X1_train)
X_MLP_train = scaler.transform(X1_train)
X_MLP_test = scaler.transform(X1_test)

# Pour éviter que certains appels plantent, nous avons augmenté le nb maximum d'itérations à 2000, rouler les
# 81 boucles prend beaucoup de temps, voici les mesures recceuillies sur l'une de nos machines
for i in range(1,10):
    for j in range(1,10):
        clf_mlp = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(i, j), random_state=1, max_iter=2000)
        clf_mlp.fit(X_MLP_train, y1_train)
        y_pred = clf_mlp.predict(X_MLP_test) # prédictions à partir du modèle sur les données test
        print('The accuracy of Multi Layered Perceptron ' +
              'when using X1 and ' + str(i)+ ' layers of ' + str(j) + ' hidden neurons is :' + str(accuracy_score(y1_test, y_pred)))

# The accuracy of Random Forest with a features window of size 1 is :0.8427109974424553
# The accuracy of Random Forest with a features window of size 2 is :0.809462915601023
# The accuracy of Random Forest with a features window of size 3 is :0.7992327365728901
# The accuracy of Random Forest with a features window of size 4 is :0.7928388746803069
# The accuracy of Random Forest when using the whole sentence is :0.6227621483375959

# The accuracy of Naive Bayes when using a features window of size 1 is :0.8478260869565217

# The accuracy of SVM when using a features window of size 1 is :0.8452685421994884

# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 1 hidden neurons is :0.5664961636828645
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 2 hidden neurons is :0.789002557544757
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 3 hidden neurons is :0.7276214833759591
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 4 hidden neurons is :0.7225063938618926
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 5 hidden neurons is :0.7915601023017903
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 6 hidden neurons is :0.7557544757033248
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 7 hidden neurons is :0.7634271099744245
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 8 hidden neurons is :0.7838874680306905
# The accuracy of Multi Layered Perceptron when using X1 and 1 layers of 9 hidden neurons is :0.7710997442455243
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 1 hidden neurons is :0.7800511508951407
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 2 hidden neurons is :0.7020460358056266
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 3 hidden neurons is :0.8107416879795396
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 4 hidden neurons is :0.8375959079283888
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 5 hidden neurons is :0.8478260869565217
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 6 hidden neurons is :0.717391304347826
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 7 hidden neurons is :0.8478260869565217
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 8 hidden neurons is :0.7902813299232737
# The accuracy of Multi Layered Perceptron when using X1 and 2 layers of 9 hidden neurons is :0.8439897698209718
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 1 hidden neurons is :0.7225063938618926
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 2 hidden neurons is :0.8324808184143222
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 3 hidden neurons is :0.8388746803069054
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 4 hidden neurons is :0.8491048593350383
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 5 hidden neurons is :0.8312020460358056
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 6 hidden neurons is :0.829923273657289
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 7 hidden neurons is :0.8836317135549873
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 8 hidden neurons is :0.8542199488491049
# The accuracy of Multi Layered Perceptron when using X1 and 3 layers of 9 hidden neurons is :0.8363171355498721
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 1 hidden neurons is :0.6560102301790282
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 2 hidden neurons is :0.850383631713555
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 3 hidden neurons is :0.8363171355498721
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 4 hidden neurons is :0.7953964194373402
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 5 hidden neurons is :0.8478260869565217
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 6 hidden neurons is :0.8529411764705882
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 7 hidden neurons is :0.8554987212276215
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 8 hidden neurons is :0.840153452685422
# The accuracy of Multi Layered Perceptron when using X1 and 4 layers of 9 hidden neurons is :0.8375959079283888
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 1 hidden neurons is :0.7429667519181585
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 2 hidden neurons is :0.8120204603580563
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 3 hidden neurons is :0.8465473145780051
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 4 hidden neurons is :0.840153452685422
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 5 hidden neurons is :0.870843989769821
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 6 hidden neurons is :0.8529411764705882
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 7 hidden neurons is :0.8721227621483376
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 8 hidden neurons is :0.8618925831202046
# The accuracy of Multi Layered Perceptron when using X1 and 5 layers of 9 hidden neurons is :0.870843989769821
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 1 hidden neurons is :0.7736572890025576
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 2 hidden neurons is :0.8375959079283888
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 3 hidden neurons is :0.8631713554987213
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 4 hidden neurons is :0.8529411764705882
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 5 hidden neurons is :0.8171355498721228
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 6 hidden neurons is :0.860613810741688
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 7 hidden neurons is :0.8465473145780051
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 8 hidden neurons is :0.8618925831202046
# The accuracy of Multi Layered Perceptron when using X1 and 6 layers of 9 hidden neurons is :0.8721227621483376
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 1 hidden neurons is :0.710997442455243
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 2 hidden neurons is :0.8107416879795396
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 3 hidden neurons is :0.8388746803069054
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 4 hidden neurons is :0.860613810741688
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 5 hidden neurons is :0.8746803069053708
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 6 hidden neurons is :0.8593350383631714
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 7 hidden neurons is :0.870843989769821
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 8 hidden neurons is :0.8567774936061381
# The accuracy of Multi Layered Perceptron when using X1 and 7 layers of 9 hidden neurons is :0.8746803069053708
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 1 hidden neurons is :0.7953964194373402
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 2 hidden neurons is :0.8286445012787724
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 3 hidden neurons is :0.8631713554987213
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 4 hidden neurons is :0.8554987212276215
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 5 hidden neurons is :0.8516624040920716
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 6 hidden neurons is :0.860613810741688
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 7 hidden neurons is :0.8657289002557544
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 8 hidden neurons is :0.8618925831202046
# The accuracy of Multi Layered Perceptron when using X1 and 8 layers of 9 hidden neurons is :0.8746803069053708
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 1 hidden neurons is :0.6815856777493606
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 2 hidden neurons is :0.8337595907928389
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 3 hidden neurons is :0.8350383631713555
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 4 hidden neurons is :0.8721227621483376
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 5 hidden neurons is :0.8554987212276215
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 6 hidden neurons is :0.8516624040920716
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 7 hidden neurons is :0.870843989769821
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 8 hidden neurons is :0.8644501278772379
# The accuracy of Multi Layered Perceptron when using X1 and 9 layers of 9 hidden neurons is :0.8682864450127877

