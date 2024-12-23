#!/bin/bash


# save_path=./log/2-layer-10,25/SAGE/h_1024
mkdir ./log
save_path=./log/buffalo
mkdir $save_path
dataset=ogbn-products
hidden=128
epoch=6
layer=2
fanout='10,25'

for nb in  12
do
    echo "---start  $nb batches. It only trains 6 epochs to save time, it will last about 3 min.'
"
    python buffalo.py \
        --dataset $dataset \
        --selection-method products_25_backpack_bucketing \
        --num-batch $nb \
        --mem-constraint 18.1 \
        --num-layers $layer \
        --fan-out $fanout \
        --model SAGE \
        --num-hidden $hidden \
        --num-runs 1 \
        --num-epoch $epoch \
        --aggre lstm \
        --log-indent 0 \
        --lr 1e-3 \
    > "${save_path}/nb_${nb}.log"
done

