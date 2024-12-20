To denote time comparison between block generation time of Buffalo and Betty, we use the figure 11 OGBN-products as an example: 2-layer GraphSAGE + LSTM aggregator ogbn-products with 12 micro-batches.  

After running `./run_buffalo_ogbn_products.sh` and `./run_betty_ogbn_products.sh`, you can get the results at the end of these files `log/buffalo/nb_12.log` and `log/betty/2-layer-fo-10,25-sage-lstm-h-128-batch-12-gp-REG.log`.  

ogbn-products GraphSAGE fan-out 10, 25 hidden 128  
| Time Comparison         | Betty       | Buffalo     |  
|-------------------------|-------------|-------------|  
| Buffalo scheduling       | -           | 2.69        |  
| REG construction         | 70.27       | -           |  
| Metis partition          | 21.50       | -           |  
| Connection check         | 13.66       | 8.17        |  
| Block construction       | 25.30       | 4.29        |  
| Data loading             | 1.43        | 3.52        |  
| Training time on GPU     | 2.37        | 2.28        |  
|-------------------------|-------------|-------------|  
| **AVG end-to-end time**  | 144.61      | 21.83       |
  

Since the machines used differ from those in the paper where the data was collected, the exact times may vary from Figure 11. However, the scale remains consistent, showing that Buffalo can reduce the end-to-end time by 83% compared to Betty in this example.