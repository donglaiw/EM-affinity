import torch
from torch.autograd import Variable
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
# 1. pruner
class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.setup()
    
    def setup(self):
        raise NotImplementedError("Need to implement setup !")

    def forward(self,x):
        raise NotImplementedError("Need to implement forward !")

    def add_conv(layer, activation_index):    
        # add one conv layer
        self.activation_to_layer[activation_index] = layer
        self.filter_ranks[activation_index] = torch.FloatTensor(layer.weight.size(0)).zero_().cuda()
        activation_index += 1
        return activation_index

	def compute_rank(self, grad):
        # reverse forward order
		activation_index = self.num_conv - self.grad_index - 1
		activation = self.activations[activation_index]
		values = torch.sum((activation * grad), dim = 0).sum(dim=2).sum(dim=3)[0, :, 0, 0].data
		# Normalize the rank by the filter dimensions
		values = values / (activation.size(0) * activation.size(2) * activation.size(3))
		self.filter_ranks[activation_index] += values
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v.cpu()

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)
		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i
		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))
		return filters_to_prune				

class FilterPrunnerVGG(FilterPrunner):
    # pruner for VGG-Imagenet
	def __init__(self, model):
        super(FilterPrunnerVGG, self).__init__(model)
 
    def add_conv_seq(module_items, activation_index):    
        for layer, (name, module) in enumerate(module_items):
            if 'Conv' in module.__str__(): # prune conv
                activation_index = add_conv(layer, activation_index)
        return activation_index

    def setup(self):
		self.filter_ranks = {}
		self.activation_to_layer = {}
        self.num_conv = add_conv_seq(self.model.features._modules.items())

	def forward(self, x):
		self.activations = [None]*self.num_conv
		self.grad_index = 0
		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
                # add backward hook
		    	x.register_hook(self.compute_rank)
		        self.activations[activation_index] = x
		        activation_index += 1
		return self.model.classifier(x.view(x.size(0), -1))

class FilterPrunnerUnet3D(FilterPrunner):
    # pruner for Unet
	def __init__(self, model):
        super(FilterPrunnerUnet3D, self).__init__(model)
 
    def add_conv_seq(module_items, activation_index):    
        # add conv seqs
        for layer, (name, module) in enumerate(module_items):
            if hasattr(module, 'cbr'):
                activation_index = add_conv(module.cbr._modules['0'], activation_index)
        return activation_index
   
    def setup(self):
		self.filter_ranks = {}
		self.activation_to_layer = {}
		activation_index = 0
        # add backward hook
        b1_names = model._modules.keys()
        for b1_name in b1_names: # down, center, up, final
            if b1_name in ['center', 'final']:
                activation_index = add_conv_seq(model._modules[b1_name]._modules['convs']._modules.items(),activation_index)
            else:
                b2_names = model._modules[b1_name]._modules.keys() 
                for b2_name in b2_names: # each layer: '0', '1'
                    b3_names = model._modules[b1_name]._modules[b2_name]._modules.keys()
                    for b3_name in b3_names: # up, up_conv, conv
                        if 'conv' in b3_name:
                            activation_index = add_conv_seq(model._modules[b1_name]._modules[b2_name]._modules[b3_name]._modules['convs']._modules.items(),activation_index)
                        elif b3_name=='up':
                            activation_index = add_conv(model._modules[b1_name]._modules[b2_name]._modules['up'], activation_index)
        self.num_conv = activation_index

	def forward(self, x):
		self.activations = [None]*self.num_conv
		self.grad_index = 0
		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
		    	x.register_hook(self.compute_rank)
		        self.activations[activation_index] = x
		        activation_index += 1
		return self.model.classifier(x.view(x.size(0), -1))


class PrunningFineTuner_unet3D:
	def __init__(self, train_data_loader, model, criterion):
		self.train_data_loader = train_data_loader
		self.model = model
		self.criterion = criterion
		self.prunner = FilterPrunner(self.model) 
		self.model.train()

	def train(self, optimizer = None, epoches = 10):
		if optimizer is None:
			optimizer = \
				optim.SGD(model.classifier.parameters(), 
					lr=0.0001, momentum=0.9)

		for i in range(epoches):
			print "Epoch: ", i
			self.train_epoch(optimizer)
		print "Finished fine tuning."
		
	def train_batch(self, optimizer, batch, label, rank_filters):
		self.model.zero_grad()
		input = Variable(batch)
		if rank_filters:
			output = self.prunner.forward(input)
			self.criterion(output, Variable(label)).backward()
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for batch, label in self.train_data_loader:
			self.train_batch(optimizer, batch.cuda(), label.cuda(), rank_filters)

	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()
		self.train_epoch(rank_filters = True)
		self.prunner.normalize_ranks_per_layer()
		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def prune(self):
		#Get the accuracy before prunning
		self.test()
		self.model.train()
		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True
		number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = 512
		iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
		iterations = int(iterations * 2.0 / 3)
		print "Number of prunning iterations to reduce 67% filters", iterations
		for _ in range(iterations):
			print "Ranking filters.. "
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 
			print "Layers that will be prunned", layers_prunned
			print "Prunning filters.. "
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				model = prune_vgg16_conv_layer(model, layer_index, filter_index)
			self.model = model.cuda()
			message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
			print "Filters prunned", str(message)
			self.test()
			print "Fine tuning to recover from prunning iteration."
			optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
			self.train(optimizer, epoches = 10)

		print "Finished. Going to fine tune the model a bit more"
		self.train(optimizer, epoches = 15)
		torch.save(model, "model_prunned")
