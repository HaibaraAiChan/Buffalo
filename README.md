# Buffalo: Enabling Large-Scale GNN Training via Memory-Efficient Bucketization  



## install requirements:
 The framework of Buffalo is developed upon DGL(pytorch backend)  
 We use Ubuntu 20.04, CUDA 12.3 (it's also compatible with Ubuntu18.04, CUDA 11.2, the package version you need to install are denoted in install_requirements.sh).  
 The requirements:  pytorch >= 1.9, DGL >= 0.8

`bash install_requirements.sh`.  

## Our main contributions: 
Buffalo introduces a system addressing the bucket explosion and enabling load balancing between graph partitions for GNN training. 

- Buffalo scheduling is implemented in  
```python
~/Buffalo/pytorch/bucketing/bucket_partitioner.py  
```
- memory-aware partitioning implementation is based on memory estimation, details are in  
```python 
~/Buffalo/pytorch/bucketing/bucket_dataloader.py  
```





   






