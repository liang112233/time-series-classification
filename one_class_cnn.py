import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


from sklearn.model_selection import KFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split



from tensorflow import keras
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

from keras.optimizers import Adam
from keras.models import load_model, save_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.callbacks import ModelCheckpoint





gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])


seq_len = 29999 # each dataset has 2999 samples
nb_epochs = 100

num_class =37
num_files_each_class = 100


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
sequences1 = list()



for j in range(1,num_class+1):
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


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(label)
encoded_label = encoder.transform(label)
# print('shape, encoded_label',encoded_label.shape, encoded_label)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_label = np_utils.to_categorical(encoded_label)

# print('label shape',dummy_label.shape)


train_data = np.reshape(sequences, (num_class*num_files_each_class, seq_len))



#
x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size = 0.2, random_state = 0)
nb_classes = len(np.unique(y_test))
# batch_size = min(x_train.shape[0]/10, 16)
batch_size = 16



x_train_mean = x_train.mean()
x_train_std = x_train.std()
x_train = (x_train - x_train_mean) / (x_train_std)

x_test = (x_test - x_train_mean) / (x_train_std)
x_train = x_train.reshape(x_train.shape + (1, 1,))
x_test = x_test.reshape(x_test.shape + (1, 1,))

y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

Y_train = keras.utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.to_categorical(y_test, nb_classes)


x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)

#    drop_out = Dropout(0.2)(conv1)
conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

#    drop_out = Dropout(0.2)(conv2)
conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

full = keras.layers.GlobalAveragePooling2D()(conv3)
out = keras.layers.Dense(num_class,activation='sigmoid')(full)



model = keras.models.Model(inputs=x, outputs=out)

optimizer = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
                 verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
#
# # # hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
# # #                  verbose=1, validation_data=(x_test, Y_test), callbacks=[reduce_lr])
# # # Print the testing results which has the lowest training loss.
# # log = pd.DataFrame(hist.history)
# # print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['accuracy'])
#
#
filepath = './saved_model'
# tf.saved_model.save(model, filepath)

# # # # #loading the model and checking accuracy on the test data
# model = load_model(filepath, compile = True)
#

#
# # Evaluate the model on the test data using `evaluate`
# print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
#
# # Generate predictions (probabilities -- the output of the last layer)
# # # on new data using `predict`
# # print("Generate predictions for 3 samples")
# predictions = model.predict(x_test)
# print("predictions:", predictions)


for i in range(1, 21):
    file_path = '/home/liang/PycharmProjects/time-series-classification/data/1/' + str(i)
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

    sequences1.append(pkgPower)
print("load data finished")
sequences1= np.array(sequences1)

# print('sequences1',sequences1)


x_test_final = np.reshape(sequences1, (20, seq_len,-1))

# np.reshape(sequences, (num_class_dataset*num_files_each_class, seq_len))

x_test_final = (x_test_final - x_train_mean) / (x_train_std)

# x_test_final = (x_test_final - x_test_final.min()) / (x_test_final.max() - x_test_final.min()) * (nb_classes - 1)

# # Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
# results = model.evaluate(x_test, y_test, batch_size=128)
# print("test loss, test acc:", results)
# Generate predictions (probabilities -- the output of the last layer)
# # on new data using `predict`
# print("Generate predictions for 3 samples")
predictions = model.predict(x_test_final)
print("predictions:", predictions)

