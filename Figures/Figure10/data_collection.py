import os
import numpy as np
import pandas as pd
from statistics import mean
import argparse
import sys

# sys.path.insert(0,'..')
# sys.path.insert(0,'../..')

def get_fan_out(filename):
	fan_out=filename.split('-')[3]
	# print(fan_out)
	return fan_out
def get_num_batch(filename):
	nb=filename.split('-')[9]
	# print(nb)
	return nb

def colored(r, g, b, text):
	return "\033[38;2;{};{};{}m{} \033[38;2;255;255;255m".format(r, g, b, text)



def avg_epoch_time(filename, args):
	
	epoch_times = []
	cuda_max_mem=[]
	f = open(filename,'r')
	for line in f:
		line = line.strip()
		
		if line.startswith("Total (block generation + training)time/epoch"):
			# print(line)
			# print(line.split())
			epoch_times.append(float(line.split()[-1]))
		
		if line.startswith("Max Memory Allocated"):
			# print('line.split()[-2]', line.split()[-2])
			cuda_max_mem.append(float(line.split()[-2]))

	f.close()

	print('\tend to end time per epoch (sec)',epoch_times[-1])
	print('\tcuda max memory consumption(GB) ',cuda_max_mem[-1])

		
		


				
def betty_data_collection( path_2, args):
    nb_folder_list=[]
    for f_item in os.listdir(path_2):
        if 'batch-' in f_item:
            nb=f_item.split('-')[9]
            nb_folder_list.append(int(nb))
    nb_folder_list.sort()
    nb_folder_list=['2-layer-fo-10,25-sage-lstm-h-1024-batch-'+str(i)+'-gp-REG.log' for i in nb_folder_list]

    
    for filename in nb_folder_list:
        f_ = os.path.join(path_2, filename)
        fan_out=get_fan_out(filename)
        nb=get_num_batch(filename)
        # import pdb; pdb.set_trace()
        print('number of batches:',nb)
        # f_='./log/arxiv/betty/2-layer-fo-10,25-sage-lstm-h-1024-batch-4-gp-REG.log'
        avg_epoch_time(f_,args)
    
def buffalo_data_collection( path_3, args):
    nb_folder_list=[]
    for f_item in os.listdir(path_3):
        # print(f_item)
        # print(f_item.split('_'))
        nb=f_item.split('_')[1]
        # print(f_item.split('_'))
        nb_folder_list.append(int(nb[:-4]))
        # print(nb_folder_list)
    nb_folder_list.sort()
    nb_folder_list=['nb_'+str(i)+'.log' for i in nb_folder_list]

    
    for filename in nb_folder_list:
        f_ = os.path.join(path_3, filename)
        # print(filename)
        
        nb=filename.split('_')[1][:-4]
        # import pdb; pdb.set_trace()
        print('number of batches:',nb)
        # f_='./log/arxiv/betty/2-layer-fo-10,25-sage-lstm-h-1024-batch-4-gp-REG.log'
        buffalo_avg_epoch_time(f_,args)
    
def buffalo_avg_epoch_time(filename, args):
	
	epoch_times = []
	cuda_max_mem=[]
	f = open(filename,'r')
	for line in f:
		line = line.strip()
		
		if line.startswith("epoch_time avg"):
			# print(line)
			# print(line.split())
			epoch_times.append(float(line.split()[-1]))
		
		if line.startswith("Max Memory Allocated"):
			# print('line.split()[-2]', line.split()[-2])
			cuda_max_mem.append(float(line.split()[-2]))

	f.close()

	print('\tend to end time per epoch (sec)',epoch_times[-1])
	print('\tcuda max memory consumption(GB) ',cuda_max_mem[-1])

if __name__=='__main__':
	
	print("computation info data collection start ...... " )
	argparser = argparse.ArgumentParser("info collection")
	# argparser.add_argument('--file', type=str, default='cora')
	# argparser.add_argument('--file', type=str, default='ogbn-products')
	argparser.add_argument('--file', type=str, default='ogbn-arxiv')
	argparser.add_argument('--model', type=str, default='sage')
	# argparser.add_argument('--aggre', type=str, default='mean')
	argparser.add_argument('--aggre', type=str, default='lstm')
	argparser.add_argument('--num-layers', type=int, default=2)

	argparser.add_argument('--hidden', type=int, default=1024)

	argparser.add_argument('--selection-method', type=str, default='REG')
	argparser.add_argument('--eval',type=bool, default=False)
	argparser.add_argument('--epoch-ComputeEfficiency', type=bool, default=False)
	argparser.add_argument('--epoch-PureTrainComputeEfficiency', type=bool, default=True)
	argparser.add_argument('--save-path',type=str, default='./')
	args = argparser.parse_args()
	print('=-=-'*20)
	print('betty result: ')
	path_2 = './log/arxiv/betty'
	betty_data_collection( path_2, args)	
	print('=-=-'*20)
	print('buffalo result: ')

	path_3 = './log/arxiv/buffalo'
	buffalo_data_collection( path_3, args)	





