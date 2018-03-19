import sys
import numpy as np
import h5py

def getLoss(fn):
    a=open(fn,'r')
    line = a.readline()
    loss = np.zeros((300000))
    lid=0
    while line:
        if 'loss=' in line:
            loss[lid] = float(line[line.find('=')+1:line.find('=')+5])
            lid+=1
        line = a.readline()
    a.close()
    return loss[:lid]

def getVI(fn):
    return [float(x) for x in open(fn,'r').readlines()[5].split(',')]

def getAvg(rr, avg_step):
    num_avg = int(np.floor(len(rr)/float(avg_step)))
    rr_avg = rr[:num_avg*avg_step].reshape((num_avg,avg_step)).mean(axis=1)
    return rr_avg

def plot_result(inN, step=100, do_plot=True, outN=None):
    import csv
    with open(inN, 'rb') as logs:
       reader = csv.reader(logs, delimiter = ' ')
       rows = list(reader)
       x = [int(row[1][:-1]) for row in rows]
       ytrain = np.array([float(row[2].split('=')[1]) for row in rows])
       ytest = np.array([float(row[3].split('=')[1]) for row in rows])
       ytrain = np.mean(ytrain[:(ytrain.shape[0]/step)*step].reshape(-1, step), axis=1)
       ytest = np.mean(ytest[:(ytest.shape[0]/step)*step].reshape(-1, step), axis=1)
       if do_plot:
           import matplotlib as mpl
           if outN is not None:
               mpl.use('Agg')
           import matplotlib.pyplot as plt
           plt.cla()
           plt.plot(ytrain)
           plt.plot(ytest)
           plt.legend(['Training loss', 'Testing loss'])
           if outN is not None:
               plt.savefig(outN)
    return ytrain, ytest
