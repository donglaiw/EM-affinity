#!/bin/python
import os
import argparse
from subprocess import check_output, call

n_volume_pretrain = 20000
n_volume_total = 200000
n_max_gpus = 5
n_cpus = 16
per_gpu = 2

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, help='The identifiant of the run')
parser.add_argument('--arch', type=str, default='0-0@0@0-0-0@0', help='The architecture of the network')
parser.add_argument('--bn', type=int, help='Whether to use BN or not')
parser.add_argument('--start-gpu', type=int, help='The first visible GPU to consider')
args = parser.parse_args()

# Get GPUs
stats = check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv']).split('\n')[1 + args.start_gpu:-1]
stats = [[int(x) for x in stat.split() if x.isdigit()] for stat in stats]
gpus = [i + args.start_gpu for i,u in enumerate(stats) if u[0]==0 and u[1]<=2]
gpus = [gpu for i, gpu in enumerate(gpus) if i < n_max_gpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,gpus))
n_gpus = len(gpus)

# Set batch
batch = per_gpu * n_gpus

# Output directories
prefix = 'results/{}/'.format(args.id)

# Actual training
snapshot = '{}/volume_{}.pth'.format(prefix, n_volume_pretrain)
call(map(str, ['python', 'E_train.py', '-a', args.arch, '-l', '0', '-lw', '2', '--volume-total', n_volume_pretrain, '--volume-save', '5000', '-o', prefix, '-bn', args.bn, '-g', n_gpus, '-b', batch, '-c', n_cpus]))
call(map(str, ['python', 'E_train.py', '-a', args.arch, '-l', '1', '-lw', '0.5', '--volume-total', n_volume_total - n_volume_pretrain, '--volume-save', '50000', '-o', prefix, '-bn', args.bn, '-g', n_gpus, '-b', batch, '-c', n_cpus, '-s', snapshot]))

# Evaluation
snapshot = '{}/volume_{}.pth'.format(prefix, n_volume_total - 1)
affinity = '{}/affinity.h5'.format(prefix)
volume = '/n/coxfs01/donglai/malis_trans/data/ecs-3d/ecs-gt-4x6x6/'
call(map(str, ['python', 'E_test.py', '-a', args.arch, '-s', snapshot, '-b', batch, '-g', n_gpus, '-c', n_cpus, '-o', affinity, '-bn', args.bn, '-i', volume]))
call(map(str, ['python', 'E_test.py', '-a', args.arch, '-bn', args.bn, '-t', 1, '-o', affinity, '-b', batch, '-c', n_cpus, '-l', 1, '-lw', 0.5, '-i', volume]))
