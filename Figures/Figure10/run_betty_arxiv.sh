#!/bin/bash


File=Betty_micro_batch_train.py

Data=ogbn-arxiv

model=sage
seed=1236 
setseed=True
GPUmem=True
load_full_batch=True
lr=0.01
dropout=0.5

run=1
epoch=5
logIndent=0

num_batch=(4 6 8 16 32)
# num_batch=(32)
pMethodList=(REG)

num_re_partition=(0)
re_partition_method=random


layersList=(2)
fan_out_list=(10,25)

hiddenList=(1024)
AggreList=(lstm)

# mkdir ./log
# mkdir ./log/arxiv
savePath=./log/arxiv/betty/
# mkdir $savePath
mkdir -p $savePath 
echo '---start Betty micro batch train: lstm aggregator. '
echo '---It trains for only 5 epochs with different batch sizes to save time, taking approximately 5 minutes for each batch size, totaling around 25 minutes.'
for Aggre in ${AggreList[@]};do

	for pMethod in ${pMethodList[@]};do
			for layers in ${layersList[@]};do
				for hidden in ${hiddenList[@]};do
					for fan_out in ${fan_out_list[@]};do
						for nb in ${num_batch[@]};do
							echo "num_batche: $nb"
							for rep in ${num_re_partition[@]};do
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
