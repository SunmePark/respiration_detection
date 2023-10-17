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
    model.add(layers.Dense(3, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.2)))

    return model





DATA = []
for num in range(1,371):
    data = pd.read_csv('./data/dataset_seg_{}.csv'.format(num))['ObjectMovement']
    DATA.append(np.array(data))

X = np.array(DATA).reshape(-1,64,1)
y = np.loadtxt('./data/label.csv', delimiter=',')
y = list(y)
y.extend([2.0]*30)
y = np.array(y)

# conf_total = np.zeros([2,2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y,  random_state=1004)

X_train_train, X_val, y_train_train, y_val = train_test_split(X_train,  y_train, test_size=0.2, shuffle=True, stratify= y_train,  random_state=1004)

print(X_train_train.shape,X_val.shape,X_test.shape)

model = model_lstm()

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt.Adam(learning_rate=0.001),
              metrics=['sparse_categorical_accuracy'])

print(model.summary())

model_path = './respiration2.hdf5'

callback_func = [EarlyStopping(monitor='val_loss', patience=20),
                ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)]

history = model.fit(X_train_train, y_train_train, validation_data=(X_val, y_val), batch_size=8, epochs=200,
                    verbose=2, callbacks=callback_func)

# y_vloss = history.history['val_loss']
# y_loss = history.history['loss']
#
# x_len = np.arange(len(y_loss))
#
# plt.figure()
# plt.plot(x_len, y_vloss, marker='.', c="red", label='testset_loss')
# plt.plot(x_len, y_loss, marker='.', c="blue", label='trainset_loss')
#
# plt.legend(loc='upper right')
# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


scores = model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict_on_batch(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

conf = confusion_matrix(y_test, y_pred_class)
print(conf)

print(scores[1])

