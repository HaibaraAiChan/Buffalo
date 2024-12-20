
 
To denote  Time comparison between block generation time of Buffalo and Betty.
We use the figure 12 (b) as an example:  2-layer GraphSAGE + LSTM aggregator ogbn-products with different number of batches.  

After `./run_buffalo_ogbn_products.sh`and `./run_betty_ogbn_products.sh` you can get the result of  16, 24, 32 micro batches.
in folder `log/products/buffalo`  and `log/products/betty`
  

ogbn-products GraphSAGE fan-out 10,25 hidden 128

|    |     16 Micro batches |     24 Micro batches |    32   Micro batches |
|-------------------------------------------------|------------:|-------------:|-------------:|
| betty:average block generation time per epoch   |       44.74 |    51.16 |     50.1 |
| buffalo:average block generation time per epoch |     11.21 |     12.51  |     14.74 |

log/products/betty/:
when number of micro-batches equals 16: at the end of file in `2-layer-fo-10,25-sage-lstm-h-128-batch-16-gp-REG.log`  
average block generation time includes:  
connection checking time:  16.63
block construction  time  28.11


log/products/buffalo/
when number of batches equals 16: at the end of file `nb_16.log`   
average block generation time includes:  
---connection_check time avg  7.19   
---block_construction time avg  4.02   
total 11.21 seconds  

The exact times you run may vary from the above table. However, the scale between betty and buffalo remains consistent.
