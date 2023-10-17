import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


X_train = np.load('./데이터/X_train.npy')
X_test = np.load('./데이터/X_test.npy')
y_train = np.load('./데이터/y_train.npy')
y_test = np.load('./데이터/y_test.npy')


model = load_model("respiration.hdf5")

scores = model.evaluate(X_test, y_test, verbose=2)

y_pred = model.predict_on_batch(X_test)
y_pred_class = np.round(y_pred)

conf = confusion_matrix(y_test, y_pred_class)

print(conf)
