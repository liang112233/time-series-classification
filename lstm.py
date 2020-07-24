import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score



from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


seq_len = 10000 # each dataset has 2999 samples



# def getpower(energy):
#   power = []
#   diff = 0
#   for i in range(0, len(energy) - 1):
#     j = energy[i + 1] - energy[i]
#     if j > 0:
#       diff = j
#     power.append(diff);
#   return power
#

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

for j in range(1,6):
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





# print('shape',sequences[0].shape)
# plt.plot(power)


labels = np.loadtxt('/home/liang/PycharmProjects/time-series-classification/data/target')
label = labels[:,1]

# print('shape,label', label.shape, label)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(label)
encoded_label = encoder.transform(label)
# print('shape, encoded_label',encoded_label.shape, encoded_label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_label = np_utils.to_categorical(encoded_label)

# print('label shape',dummy_label.shape)


kfold = KFold(n_splits=10, shuffle=True)

train_data = np.reshape(sequences, (500, seq_len, -1))



def baseline_model():
    #create model
    model = Sequential()
    # model.add(LSTM(units=512, input_shape=(200, 1)))  #batch size 5
    model.add(LSTM(256, input_shape=(seq_len, 1)))

    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


estimator = KerasClassifier(build_fn=baseline_model, epochs=50, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, shuffle=True)

# results = cross_val_score(estimator, sequences, dummy_label, cv=kfold)
results = cross_val_score(estimator, train_data, dummy_label, cv=kfold)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#
# #
# # #
# # # # adam = Adam(lr=0.001)
# # #
# # # # model_filename = "test-Epoch-{epoch:02d}"
# # # # checkpoint_path = os.path.join('models/', model_filename)
# # #
# # # # chk = ModelCheckpoint(
# # # #     filepath='best_model.pkl',
# # # #     monitor='val_accuracy',
# # # #     verbose=1,
# # # #     save_best_only=True,
# # # #     save_weights_only=False,
# # # #     mode='max')
# # # #
# # # # # chk = ModelCheckpoint('best_model.pkl', monitor='val_acc', save_best_only=True, mode='max', verbose=1)
# # # # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
# # # # model.fit(train, train_target, epochs=200, batch_size=128, callbacks=[chk], validation_data=(validation,validation_target))
# # #

