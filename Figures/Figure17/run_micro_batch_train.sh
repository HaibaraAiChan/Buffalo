#!/bin/bash


File=buffalo.py
Data=ogbn-arxiv
pMethodList=(25_backpack_arxiv_bucketing)

model=SAGE
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.001
dropout=0.5

run=1
epoch=10
logIndent=0
cuda_mem_constraint=18.1 # 18.1, 24 GB GPU memory, there is about 6 GB  nvidia-smi memory cost

num_batch=(1 2 4 8) 
num_re_partition=(0)

layersList=(2)
fan_out_list=(10,25)

hiddenList=(128 )
AggreList=(lstm )


mkdir ./log

save_path=./log
mkdir $save_path

for Aggre in ${AggreList[@]}
do      
	for pMethod in ${pMethodList[@]}
	do      
			for layers in ${layersList[@]}
			do      
				for hidden in ${hiddenList[@]}
				do
					for fan_out in ${fan_out_list[@]}
					do
						
						for nb in ${num_batch[@]}
						do
							
							for rep in ${num_re_partition[@]}
							do
							echo 'number of batches equals '${nb}
							python $File \
								--dataset $Data \
								--selection-method $pMethod \
								--num-batch $nb \
								--mem-constraint $cuda_mem_constraint \
								--num-layers $layers \
								--fan-out $fan_out \
								--num-hidden $hidden \
								--num-runs $run \
								--num-epoch $epoch \
								--aggre $Aggre \
								--log-indent $logIndent \
								--lr $lr \
							> ${save_path}/${layers}_layer_aggre_${Aggre}_batch_${nb}.log
							done
						done
					done
				done
			done
		
	done
done
