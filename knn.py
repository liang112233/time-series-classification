import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn import metrics



from math import sqrt

seq_len = 29999 # each dataset has 2999 samples



def getpower(data):
  power = []
  diff = 0
  for i in range(0, len(data) - 1):
    j = int(data[i + 1]) - int(data[i])
    if j != 0:
      diff = j
    power.append(diff)
  return power


#
path = '/home/liang/PycharmProjects/time-series-classification/data/'
sequences = list()

for j in range(1,38):
    file_path1 = path + str(j)+'/'
    for i in range(1,101):
        file_path = file_path1 + str(i)
        # print(file_path)
        # energy=np.loadtxt(file_path)
        f = open(file_path, "r")
        lines = f.readlines()
        pkgEnergy = []
        cpuEnergy = []
        gpuEnergy = []
        dramEnergy = []
        for l in lines:
            v = l.split('\t')
            pkgEnergy.append(v[0])
            cpuEnergy.append(v[1])
            gpuEnergy.append(v[2])
            dramEnergy.append(v[3])
        pkgPower = getpower(pkgEnergy)[0:seq_len]
        cpuPower = getpower(cpuEnergy)
        gpuPower = getpower(gpuEnergy)
        dramPower = getpower(dramEnergy)


        # power = getpower(df)
        # power = np.diff(energy, axis = 0)
        sequences.append(pkgPower)
print("load data finished")
sequences= np.array(sequences)





labels = np.loadtxt('/home/liang/PycharmProjects/time-series-classification/data/target_cp')
label = labels[:,1]

# print('shape,label', label.shape, label)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(label)
encoded_label = encoder.transform(label)




#
# # calculate the Euclidean distance between two vectors
# def euclidean_distance(row1, row2):
#     distance = 0.0
#     for i in range(len(row1) - 1):
#         distance += (row1[i] - row2[i]) ** 2
#     return sqrt(distance)





# X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_label, test_size=0.3) # 95% training and 5% test

kf = KFold(n_splits=5)



#Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)

# #Train the model using the training sets
# knn.fit(X_train, y_train)
#
# #Predict the response for test dataset
# y_pred = knn.predict(X_test)
scores = cross_val_score(knn, sequences, label, cv=10, scoring='accuracy')
print('Accuracies',scores)

# # Model Accuracy, how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print('Accuracy mean',scores.mean())

