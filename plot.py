import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

with open('results/with-bn/log_malis_0.0001_0.5volume_20000.txt', 'rb') as logs:
     reader = csv.reader(logs, delimiter = ' ')
     rows = list(reader)
     x = [int(row[1][:-1]) for row in rows]
     ytrain = np.array([float(row[2].split('=')[1]) for row in rows])
     ytest = np.array([float(row[3].split('=')[1]) for row in rows])
     ytrain = np.mean(ytrain.reshape(-1, 100), axis=1)
     ytest = np.mean(ytest.reshape(-1, 100), axis=1)
     plt.plot(ytrain)
     plt.plot(ytest)
     plt.legend(['Training loss', 'Testing loss'])
     plt.savefig('plot.png')
