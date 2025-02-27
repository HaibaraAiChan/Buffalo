
from dgl.data.utils import save_graphs
import numpy as np
from statistics import mean

import gc


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset

import random
import time
import dgl.function as fn

import tracemalloc


import pickle

import os 
import sys
sys.path.insert(0,'..')
sys.path.insert(0,'../micro_batch_train')
sys.path.insert(0,'../models')
sys.path.insert(0,'../utils')
from memory_usage import see_memory_usage, nvidia_smi_usage
import argparse


def set_seed(args):
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.device >= 0:
		torch.cuda.manual_seed_all(args.seed)
		torch.cuda.manual_seed(args.seed)
		torch.backends.cudnn.enabled = False
		torch.backends.cudnn.deterministic = True
		dgl.seed(args.seed)
		dgl.random.seed(args.seed)

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        # self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h






def CPU_DELTA_TIME(tic, str1):
	toc = time.time()
	print(str1 + ' spend:  {:.6f}'.format(toc - tic))
	return toc


def compute_acc(pred, labels):
	"""
	Compute the accuracy of prediction given the labels.
	"""
	labels = labels.long()
	return (torch.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeats, labels, train_nid, val_nid, test_nid, device, args):
	"""
	Evaluate the model on the validation set specified by ``val_nid``.
	g : The entire graph.
	inputs : The features of all the nodes.
	labels : The labels of all the nodes.
	val_nid : the node Ids for validation.
	device : The GPU device to evaluate on.
	"""
	# train_nid = train_nid.to(device)
	# val_nid=val_nid.to(device)
	# test_nid=test_nid.to(device)
	nfeats=nfeats.to(device)
	g=g.to(device)
	# print('device ', device)
	model.eval()
	with torch.no_grad():
		# pred = model(g=g, x=nfeats)
		pred = model.inference(g, nfeats,  args, device)
	model.train()
	
	train_acc= compute_acc(pred[train_nid], labels[train_nid].to(pred.device))
	val_acc=compute_acc(pred[val_nid], labels[val_nid].to(pred.device))
	test_acc=compute_acc(pred[test_nid], labels[test_nid].to(pred.device))
	return (train_acc, val_acc, test_acc)


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
	"""
	Extracts features and labels for a subset of nodes
	"""
	batch_inputs = nfeat[input_nodes].to(device)
	batch_labels = labels[seeds].to(device)
	return batch_inputs, batch_labels

def load_block_subtensor( blocks, device, args):
	"""
	Extracts features and labels for a subset of nodes
	"""

	if args.GPUmem:
		see_memory_usage("----------------------------------------before batch input features to device")
	batch_inputs = blocks[0].srcdata['feat'].to(device)
	if args.GPUmem:
		see_memory_usage("----------------------------------------after batch input features to device")
	batch_labels = blocks[-1].dstdata['label'].to(device)
	print(type(batch_labels))
	if args.GPUmem:
		see_memory_usage("----------------------------------------after  batch labels to device")
	return batch_inputs, batch_labels

def get_compute_num_nids(blocks):
	res=0
	for b in blocks:
		res+=len(b.srcdata['_ID'])
	return res

def get_num_edges(blocks):
	res=0
	for b in blocks:
		res+=b.num_edges()
	return res
	
def get_FL_output_num_nids(blocks):
	
	output_fl =len(blocks[0].dstdata['_ID'])
	return output_fl



#### Entry point
def run(args, device, g, dataset, model):
	if args.GPUmem:
		see_memory_usage("----------------------------------------start of run function ")
	# Unpack data
	train_nid = dataset.train_idx.to(device)
	# val_idx = dataset.val_idx.to(device)
	full_batch_size = len(train_nid)
	sampler = NeighborSampler([10, 25],  # fanout for [layer-0, layer-1]
							prefetch_node_feats=['feat'],
							prefetch_labels=['label'])
	
	
	batch_size = int(full_batch_size/args.num_batch) + (full_batch_size % args.num_batch>0)
	args.batch_size = batch_size
	

	args.num_workers = 0
	full_batch_dataloader = dgl.dataloading.NodeDataLoader(
		g,
		train_nid,
		sampler,
		device='cpu',
		batch_size=batch_size,
		shuffle=True,
		drop_last=False,
		num_workers=args.num_workers)
	if args.GPUmem:
		see_memory_usage("----------------------------------------before model to device ")


	
	loss_fcn = F.cross_entropy
	if args.GPUmem:
		see_memory_usage("----------------------------------------after model to device")
	
	dur = []
	time_block_gen=[]
	for run in range(args.num_runs):
		model.train()
		optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
		
		# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

		for epoch in range(args.num_epochs):
			print('epoch ', epoch)
			model.train()
			total_loss = 0
			see_memory_usage("----------------------------------------before dataloader")
			for step, (input_nodes, seeds, blocks) in enumerate(full_batch_dataloader):
				see_memory_usage("----------------------------------------after dataloader")
				x = blocks[0].srcdata['feat'].to(torch.device('cuda'))###############
				y = blocks[-1].dstdata['label'].to(torch.device('cuda'))
				blocks = [block.to(torch.device('cuda')) for block in blocks]#------------*
				
				see_memory_usage("----------------------------------------before model forward")

				batch_pred = model(blocks, x)#------------*
				
				see_memory_usage("----------------------------------------after model forward")
				loss = loss_fcn(batch_pred, y)#------------*
				loss.backward()#------------*
				see_memory_usage("----------------------------------------after backward")
				optimizer.step()
				optimizer.zero_grad()
				see_memory_usage("----------------------------------------after opt update")				
				total_loss += loss.item()
    
			print()
			print("Epoch {:05d} | Loss {:4f} ".format(epoch, total_loss))
				
		
    

	
def count_parameters(model):
	pytorch_total_params = sum(torch.numel(p) for p in model.parameters())
	print('total model parameters size ', pytorch_total_params)
	print('trainable parameters')
    
	for name, param in model.named_parameters():
		if param.requires_grad:
			print (name + ', '+str(param.data.shape))
	print('-'*40)
	print('un-trainable parameters')
	for name, param in model.named_parameters():
		if not param.requires_grad:
			print (name, param.data.shape)

def main():
	# get_memory("-----------------------------------------main_start***************************")
	tt = time.time()
	print("main start at this time " + str(tt))
	argparser = argparse.ArgumentParser("multi-gpu training")
	argparser.add_argument('--device', type=int, default=0,
		help="GPU device ID. Use -1 for CPU training")
	argparser.add_argument('--seed', type=int, default=1236)
	argparser.add_argument('--setseed', type=bool, default=True)
	argparser.add_argument('--GPUmem', type=bool, default=True)
	# argparser.add_argument('--load-full-batch', type=bool, default=False)
	# argparser.add_argument('--load-full-batch', type=bool, default=True)
	# argparser.add_argument('--root', type=str, default='../my_full_graph/')
	# argparser.add_argument('--dataset', type=str, default='ogbn-arxiv')
	# argparser.add_argument('--dataset', type=str, default='ogbn-mag')
	argparser.add_argument('--dataset', type=str, default='ogbn-products')
	# argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--dataset', type=str, default='cora')
	# argparser.add_argument('--dataset', type=str, default='karate')
	# argparser.add_argument('--dataset', type=str, default='reddit')
	argparser.add_argument('--aggre', type=str, default='mean')
	# argparser.add_argument('--aggre', type=str, default='pool')

	#-------------------------------------------------------------------------------------------------------
	argparser.add_argument('--num-runs', type=int, default=1)
	argparser.add_argument('--num-epochs', type=int, default=10)

	argparser.add_argument('--num-hidden', type=int, default=128)

	# argparser.add_argument('--num-layers', type=int, default=3)
	# argparser.add_argument('--fan-out', type=str, default='10,25,30')
	argparser.add_argument('--num-layers', type=int, default=2)
	argparser.add_argument('--fan-out', type=str, default='10,25')


	argparser.add_argument('--num-batch', type=int, default=1) #<---===========
	argparser.add_argument('--batch-size', type=int, default=0)

	argparser.add_argument('--log-indent', type=float, default=10)

	argparser.add_argument('--lr', type=float, default=1e-3)
	argparser.add_argument('--dropout', type=float, default=0.5)
	argparser.add_argument("--weight-decay", type=float, default=5e-4,
						help="Weight for L2 loss")
	argparser.add_argument("--eval", action='store_true', 
						help='If not set, we will only do the training part.')

	argparser.add_argument('--num-workers', type=int, default=0,
		help="Number of sampling processes. Use 0 for no extra process.")
	argparser.add_argument("--eval-batch-size", type=int, default=100000,
						help="evaluation batch size")
	argparser.add_argument("--R", type=int, default=5,
						help="number of hops")

	argparser.add_argument('--log-every', type=int, default=5)
	argparser.add_argument('--eval-every', type=int, default=5)

	args = argparser.parse_args()
	if args.setseed:
		set_seed(args)
	device = "cpu"
	if args.GPUmem:
		see_memory_usage("-----------------------------------------before load data ")
	# load and preprocess dataset
	print('Loading data')
	dataset = AsNodePredDataset(DglNodePropPredDataset(args.dataset))
	g = dataset[0]
	
	# create GraphSAGE model
	in_size = g.ndata['feat'].shape[1]
	out_size = dataset.num_classes
	model = SAGE(in_size, args.num_hidden, out_size).to(torch.device('cuda'))

	best_test = run(args, device, g, dataset, model)
	

if __name__=='__main__':
	main()


