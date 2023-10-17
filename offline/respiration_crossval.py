import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from tensorflow.keras import Sequential
from tensorflow.keras.utils import Sequence, plot_model
from tensorflow.keras import layers
from tensorflow.keras import initializers
import tensorflow.keras.optimizers as opt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftfreq, ifft



def model_lstm():
    model = Sequential()

    model.add(layers.Input(shape=(None, 1)))
    model.add(layers.LSTM(20))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid', kernel_initializer=initializers.RandomNormal(stddev=0.2)))

    return model


X = np.loadtxt('./데이터/X.csv', delimiter=',').reshape(305,64,1)
y = np.loadtxt('./데이터/y.csv', delimiter=',')


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1004)


conf_total = np.zeros([2,2])


for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train_train, X_val, y_train_train, y_val = train_test_split(X_train,  y_train, test_size=0.25, shuffle=True, stratify= y_train,  random_state=1004)

    print(X_train_train.shape,X_val.shape,X_test.shape)

    model = model_lstm()

    model.compile(loss='binary_crossentropy', optimizer=opt.Adam(learning_rate=0.001), metrics=['accuracy'])
    print(model.summary())

    model_path = './respiration_crossval.hdf5'

    callback_func = [EarlyStopping(monitor='val_loss', patience=20),
                    ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train_train, y_train_train, validation_data=(X_val, y_val), batch_size=8, epochs=200,
                        verbose=0, callbacks=callback_func)

    # y_vloss = history.history['val_loss']
    # y_loss = history.history['loss']

    # x_len = np.arange(len(y_loss))

    # plt.figure()
    # plt.plot(x_len, y_vloss, marker='.', c="red", label='testset_loss')
    # plt.plot(x_len, y_loss, marker='.', c="blue", label='trainset_loss')

    # plt.legend(loc='upper right')
    # plt.grid()
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.show()


    scores = model.evaluate(X_test, y_test, verbose=2)

    y_pred = model.predict_on_batch(X_test)
    y_pred_class = np.round(y_pred)

    conf = confusion_matrix(y_test, y_pred_class)
    print(conf)
    conf_total = conf_total + conf

print(conf_total)
print((conf_total[0][0]+conf_total[1][1])/np.sum(conf_total))

