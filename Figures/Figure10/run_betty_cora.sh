#!/bin/bash


File=Betty_micro_batch_train.py

Data=cora

model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=20
logIndent=2

num_batch=(3 4 5 6)

pMethodList=(REG)

num_re_partition=(0)
re_partition_method=random


layersList=(3)
fan_out_list=(10,25,30)

hiddenList=(2048)
AggreList=(lstm)


savePath=./log/cora/betty/

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
								wf=${layers}-layer-fo-${fan_out}-sage-${Aggre}-h-${hidden}-batch-${nb}-gp-${pMethod}.log
								echo $wf

								python $File \
								--dataset $Data \
								--aggre $Aggre \
								--seed $seed \
								--setseed $setseed \
								--GPUmem $GPUmem \
								--selection-method $pMethod \
								--re-partition-method $re_partition_method \
								--num-re-partition $rep \
								--num-batch $nb \
								--lr $lr \
								--num-runs $run \
								--num-epochs $epoch \
								--num-layers $layers \
								--num-hidden $hidden \
								--dropout $dropout \
								--fan-out $fan_out \
								--log-indent $logIndent \
								--load-full-batch True \
								> ${savePath}${wf}

							done
						done
					done
				done
			done
		
	done
done
