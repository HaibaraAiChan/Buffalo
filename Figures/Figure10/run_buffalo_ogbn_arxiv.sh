#!/bin/bash


# save_path=./log/2-layer-10,25/SAGE/h_1024
save_path=./log/arxiv/buffalo
mkdir $save_path
dataset=ogbn-arxiv
hidden=1024
layer=2
fanout='10,25'

for nb in  4 6 8 16 32 
do
    echo "---start  $nb batches"
    python buffalo.py \
        --dataset $dataset \
        --selection-method arxiv_25_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 18.1 \
        --num-layers $layer \
        --fan-out $fanout \
        --model SAGE \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch 20 \
        --aggre lstm \
        --log-indent 3 \
        --lr 1e-3 \
    > "${save_path}/nb_${nb}.log"
done

