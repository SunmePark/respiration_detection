import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

timestamp = ['2021-10-27']*64
state = [0]*64
rpm = [0]*64
objectdistance = [0]*64
signalquality = [0]*64
label = [0]*64



for i in range(1,30):

    mu = np.random.uniform(low=-0.05, high=0.05, size=(1,))
    sigma = np.random.uniform(low=0.7, high=1.3, size=(1,))
    objectmovement = np.random.normal(mu, sigma, 64)

    plt.plot(objectmovement)

    dummy = pd.DataFrame(
        {'TimeStamp':timestamp,
        'State':state,
        'RPM':rpm,
        'ObjectDistance':objectdistance,
        'ObjectMovement':objectmovement,
        'SignalQuality':signalquality,
        'Label':label})

    pd.DataFrame(dummy).to_csv('./data/dataset_seg_%d.csv' %(i+371), index=None)

