To denote time comparison between block generation time of Buffalo and Betty, we use the figure 11 OGBN-products as an example: 2-layer GraphSAGE + LSTM aggregator ogbn-products with 12 micro-batches.  

After running `./run_buffalo_ogbn_products.sh` and `./run_betty_ogbn_products.sh`, you can get the results at the end of these files `log/buffalo/nb_12.log` and `log/betty/2-layer-fo-10,25-sage-lstm-h-128-batch-12-gp-REG.log`.  

ogbn-products GraphSAGE fan-out 10, 25 hidden 128  
| Time Comparison         | Betty       | Buffalo     |  
|-------------------------|-------------|-------------|  
| Buffalo scheduling       | -           | 2.49        |  
| REG construction         | 69.07       | -           |  
| Metis partition          | 19.90       | -           |  
| Connection check         | 12.67       | 6.32        |  
| Block construction       | 23.71       | 3.80        |  
| Data loading             | 2.02        | 3.46        |  
| Training time on GPU     | 2.31        | 2.27        |  
|-------------------------|-------------|-------------|  
| **AVG end-to-end time**  | 136.26      | 19.18       |
  

Since the machines used differ from those in the paper where the data was collected, the exact times may vary from Figure 11. However, the scale remains consistent, showing that Buffalo can reduce the end-to-end time by 85% compared to Betty in this example.