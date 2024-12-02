import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

	
def read_acc_loss(filename):
    
    loss_array = []
    
    with open(filename) as f:
        # print(filename)
        # filename = './2_layer_aggre_lstm_batch_1.log'
        for line in f:
            if ('pseudo_mini_loss sum'in line.strip() )or ( 'full batch train ------ loss' in line.strip()):
                
                
                loss=line.split()[-1]
                print(loss)
                loss = float(loss)
                
                loss_array.append(loss)
                
    print(loss_array[:10])
    print(len(loss_array))
    return loss_array

def data_formalize(loss_list):
    
    len_1=len(loss_list[0])
    len_2=len(loss_list[1])
    len_4=len(loss_list[2])
    len_8=len(loss_list[3])
    len_cut = min([len_1,len_2, len_4, len_8 ])

    for i in range(4):
        loss_list[i] = loss_list[i][:len_cut]
    
    x=range(len_cut)

    return x, loss_list[0], loss_list[1], loss_list[2], loss_list[3]

def draw(loss_list):
	fig, ax = plt.subplots( figsize=(7, 4))
	# ax.set_facecolor('0.8')
	print()
    
	x, acc_1, acc_2, acc_4, acc_8,= data_formalize(loss_list)
	ax.plot(x, acc_1, '-',label='full batch train', color='orange')
	ax.plot(x, acc_2, '--',label='2 micro batch train', color='purple')
	ax.plot(x, acc_4, '-.',label='4 micro batch train', color='lime')
	ax.plot(x, acc_8, ':', label='8 micro batch train', color='red')
	
	
	ax.set(xlabel='Epoch', ylabel='Loss')
	plt.legend()
	plt.savefig('Figure17.png')
	

				
def data_collection( path, args):
	
	nb_folder_list=[]
	loss_list = []

	for f_item in os.listdir(path):
		if 'batch_' in f_item:
			print('f_item', f_item)
			nb_size=f_item.split('_')[5]
			nb_size = nb_size[:-4]
			nb_folder_list.append(int(nb_size))
			
	nb_folder_list.sort()
	nb_folder_list=['2_layer_aggre_lstm_batch_'+str(i)+'.log' for i in nb_folder_list]
	for file in nb_folder_list:
		print(loss_list)
		# import pdb; pdb.set_trace()
		res  = read_acc_loss(path+file)
		loss_list.append( res )
	print(loss_list)
	draw(loss_list )


if __name__=='__main__':
    
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='cora')
	# argparser.add_argument('--file', type=str, default='ogbn-products')
	argparser.add_argument('--file', type=str, default='ogbn-arxiv')
	argparser.add_argument('--model', type=str, default='sage')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	# argparser.add_argument('--num-layers', type=int, default=1)
	# argparser.add_argument('--hidden', type=int, default=32)
	argparser.add_argument('--hidden', type=int, default=128)
	# argparser.add_argument('--selection-method', type=str, default='range')
	# argparser.add_argument('--selection-method', type=str, default='random')
	# argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--eval',type=bool, default=False)
	argparser.add_argument('--epoch-ComputeEfficiency', type=bool, default=False)
	argparser.add_argument('--epoch-PureTrainComputeEfficiency', type=bool, default=True)
	argparser.add_argument('--save-path',type=str, default='./')
	args = argparser.parse_args()
	
	path = './log/'
	data_collection( path, args)		





