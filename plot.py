import csv
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
def plot_result(inN,outN='result.png'):
    with open(inN, 'rb') as logs:
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
         plt.savefig(outN)
