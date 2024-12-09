To denote the trend of the reduction of memory consumption and increase of training time as we increase the number of batches, we use figures 10 (d) as an example: a 2-layer GraphSAGE + LSTM aggregator on ogbn-arxiv with different numbers of batches.  

### OGBN-Arxiv GraphSAGE  

After running `./run_betty_arxiv.sh`, you can obtain the results for full batch, 4, 8, 16, and 32 micro batches in the folder `log/arxiv/betty`.  

|    |     4 Micro Batches |     6 Micro Batches |     8 Micro Batches |    16 Micro Batches |    32 Micro Batches |  
|----|----------------------|----------------------|----------------------|----------------------|----------------------|  
| Average End-to-End Time per Epoch |    54.96 |     49.52 |     48.03 |     50.47 |     50.32 |  
| CUDA Max Memory Consumption        |    21.84 |     15.21 |     12.23 |     7.72  |     4.18  |  

After running `./run_buffalo_arxiv.sh`, you can obtain the results for full batch, 4, 8, 16, and 32 micro batches in the folder `log/arxiv/buffalo`.  

|    |     4 Micro Batches |     6 Micro Batches |     8 Micro Batches |    16 Micro Batches |    32 Micro Batches |  
|----|----------------------|----------------------|----------------------|----------------------|----------------------|  
| Average End-to-End Time per Epoch |    5.18  |     5.36  |     5.60  |     6.34  |     7.36  |  
| CUDA Max Memory Consumption        |    18.39 |     15.22 |     12.11 |     7.35  |     4.76  |  

After executing `./run_betty_arxiv.sh` and `./run_buffalo_arxiv.sh`, run `python data_collection.py > time_memory.log` to see the results displayed in the tables above.

Since the machines used differ from those in the paper where the data was collected, the exact times may vary from Figure 10. However, the scale remains consistent, showing that Buffalo can significantly reduce the end-to-end time compared to Betty.


<!-- To denote the trend of the reduction of memory consumption and increase of training time as we increase the number of batches, we use figures 10 (a) and (d) as examples: a 3-layer GraphSAGE + LSTM aggregator on Cora and a 2-layer GraphSAGE + LSTM aggregator on ogbn-arxiv with different numbers of batches.   -->

<!-- ### Cora GraphSAGE  

- Fan-out: 10, 25, 30  
- Hidden units: 2048  

After running `./run_betty_cora.sh`, you can obtain the results for full batch, 4, 8, 16, and 32 micro batches in the folder `log/cora/betty`.  

|    |   Full Batch  |      2 Micro Batches |     3 Micro Batches |     4 Micro Batches |     5 Micro Batches |    6 Micro Batches |  
|----|---------------|----------------------|----------------------|----------------------|----------------------|---------------------|  
| Average End-to-End Time per Epoch |     0.75 |   1.26 |    1.60 |     1.88 |     2.16 |     2.39 |  
| CUDA Max Memory Consumption        |    2.5   |   2.39 |    2.29 |     2.17 |     2.11 |     2.09 |  

After running `./run_buffalo_cora.sh`, you can obtain the results for full batch, 4, 8, 16, and 32 micro batches in the folder `log/cora/buffalo`.  

|    |   Full Batch  |      2 Micro Batches |     3 Micro Batches |     4 Micro Batches |     5 Micro Batches |    6 Micro Batches |  
|----|---------------|----------------------|----------------------|----------------------|----------------------|---------------------|  
| Average End-to-End Time per Epoch |     0.74 |   1.19 |    1.36 |     1.78 |     2.04 |     2.22 |  
| CUDA Max Memory Consumption        |    2.5   |   2.4  |    2.21 |     2.13 |     2.07 |     2.01 |   -->

