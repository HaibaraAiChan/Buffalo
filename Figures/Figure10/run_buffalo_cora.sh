#!/bin/bash

# save_path=./log/3-layer-10,25,30/SAGE/h_2048
save_path=./log/cora/buffalo
mkdir $save_path
dataset=cora
hidden=2048
layer=3
fanout='10,25,30'

for nb in 1 2 3 4 5 6
do
    echo "---start  $nb batches"
    python buffalo.py \
        --dataset $dataset \
        --selection-method cora_30_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 7.5 \
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

