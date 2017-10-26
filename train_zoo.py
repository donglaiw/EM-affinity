#!/bin/python
import os
from subprocess import check_output, call

n_iter_pretrain = 2000
n_iter_total = 17000
n_max_gpus = 5
n_cpus = 16
per_gpu = 2
id = 0
arch = '0-0@0@1-0-0@0'
bn = 0
start_gpu = 0

# Get GPUs
stats = check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv']).split('\n')[1 + start_gpu:-1]
stats = [[int(x) for x in stat.split() if x.isdigit()] for stat in stats]
gpus = [i + start_gpu for i,u in enumerate(stats) if u==[0, 0]]
gpus = [gpu for i, gpu in enumerate(gpus) if i < n_max_gpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpus))
n_gpus = len(gpus)

# Set batch
batch = per_gpu * n_gpus

# Output directories
prefix = 'results/{}/'.format(id)

# Actual training
snapshot = '{}/iter_{}.pth'.format(prefix, n_iter_pretrain)
call(map(str, ['python', 'E_train.py', '-a', arch, '-l', '0', '-lw', '2', '--iter-total', n_iter_pretrain, '--iter-save', '500', '-o', prefix, '-bn', bn, '-g', n_gpus, '-b', batch, '-c', n_cpus]))
call(map(str, ['python', 'E_train.py', '-a', arch, '-l', '1', '-lw', '0.5', '--iter-total', n_iter_total - n_iter_pretrain, '--iter-save', '5000', '-o', prefix, '-bn', bn, '-g', n_gpus, '-b', batch, '-c', n_cpus, '-s', snapshot]))

# Evaluation
snapshot = '{}/iter_{}.pth'.format(prefix, n_iter_total - 1)
affinity = '{}/affinity.h5'.format(prefix)
volume = '/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/'
call(map(str, ['python', 'E_test.py', '-a', arch, '-s', snapshot, '-b', batch, '-g', n_gpus, '-c', n_cpus, '-o', affinity, '-bn', bn, '-i', volume]))
call(map(str, ['python', 'E_test.py', '-a', arch, '-bn', bn, '-t', 1, '-o', affinity, '-b', batch, '-c', n_cpus, '-l', 1, '-lw', 0.5, '-i', volume]))
