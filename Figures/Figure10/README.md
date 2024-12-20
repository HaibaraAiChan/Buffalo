To denote the trend of the reduction of memory consumption and increase of training time as we increase the number of batches, we use figures 10 (d) as an example: a 2-layer GraphSAGE + LSTM aggregator on ogbn-arxiv with different numbers of batches.  

### OGBN-Arxiv GraphSAGE  

After running `./run_betty_arxiv.sh`, you can obtain the results for  4, 8, 16, and 32 micro batches in the folder `log/arxiv/betty`.  

|    |     4 Micro Batches |     6 Micro Batches |     8 Micro Batches |    16 Micro Batches |    32 Micro Batches |  
|----|----------------------|----------------------|----------------------|----------------------|----------------------|  
| Average End-to-End Time per Epoch(s) |    50.13 |     47.64 |     46.18 |     45.07 |     47.28 |  
| CUDA Max Memory Consumption(GB)        |    21.79 |     14.91 |     11.99 |     7.43  |     4.20  |  

After running `./run_buffalo_arxiv.sh`, you can obtain the results for  4, 8, 16, and 32 micro batches in the folder `log/arxiv/buffalo`.  

|    |     4 Micro Batches |     6 Micro Batches |     8 Micro Batches |    16 Micro Batches |    32 Micro Batches |  
|----|----------------------|----------------------|----------------------|----------------------|----------------------|  
| Average End-to-End Time per Epoch(s) |    5.22  |     5.36  |     5.63  |     6.42  |     7.22  |  
| CUDA Max Memory Consumption(GB)        |    18.39 |     15.22 |     12.11 |     7.35  |     4.76  |  

After executing `./run_betty_arxiv.sh` and `./run_buffalo_arxiv.sh`, run `python data_collection.py > time_memory.log` to see the results displayed in the tables above.

Since the machines used differ from those in the paper where the data was collected, the exact times may vary from Figure 10(d). However, the scale remains consistent, showing that Buffalo can significantly reduce the end-to-end time compared to Betty.



<!-- 
20 epochs results  
### OGBN-Arxiv GraphSAGE  

After running `./run_betty_arxiv.sh`, you can obtain the results for  4, 8, 16, and 32 micro batches in the folder `log/arxiv/betty`.  

|    |     4 Micro Batches |     6 Micro Batches |     8 Micro Batches |    16 Micro Batches |    32 Micro Batches |  
|----|----------------------|----------------------|----------------------|----------------------|----------------------|  
| Average End-to-End Time per Epoch(s) |    54.96 |     49.52 |     48.03 |     50.47 |     50.32 |  
| CUDA Max Memory Consumption(GB)        |    21.84 |     15.21 |     12.23 |     7.72  |     4.18  |  

After running `./run_buffalo_arxiv.sh`, you can obtain the results for  4, 8, 16, and 32 micro batches in the folder `log/arxiv/buffalo`.  

|    |     4 Micro Batches |     6 Micro Batches |     8 Micro Batches |    16 Micro Batches |    32 Micro Batches |  
|----|----------------------|----------------------|----------------------|----------------------|----------------------|  
| Average End-to-End Time per Epoch(s) |    5.18  |     5.36  |     5.60  |     6.34  |     7.36  |  
| CUDA Max Memory Consumption(GB)        |    18.39 |     15.22 |     12.11 |     7.35  |     4.76  |  

After executing `./run_betty_arxiv.sh` and `./run_buffalo_arxiv.sh`, run `python data_collection.py > time_memory.log` to see the results displayed in the tables above.

Since the machines used differ from those in the paper where the data was collected, the exact times may vary from Figure 10. However, the scale remains consistent, showing that Buffalo can significantly reduce the end-to-end time compared to Betty.
 -->
