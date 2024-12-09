# Buffalo: Enabling Large-Scale GNN Training via Memory-Efficient Bucketization   



## install requirements:
 The framework of Buffalo is developed upon DGL(pytorch backend)  
 We use Ubuntu 20.04, CUDA 12.1 (the package version you need to install are denoted in install_requirements.sh).  
 The requirements:  pytorch >= 2.1, DGL >= 2.2  


`bash install_requirements.sh`.  

## Our main contributions:   
Buffalo introduces a system addressing the bucket explosion and enabling load balancing between graph partitions for GNN training.  

<!-- ###image `ubuntu_22.04_CUDA12.1_py3.10_DGL_source_modified_sampler` use the modified dgl sampler      ### -->
  
As we do optimization in DGL to speedup the block generation of Buffalo. Hence, you need install the modified DGL first. The steps are shown below. Or you can run the evalutaion in the node we provied which installed modified DGL and all requirements. 
`ssh -i passwd.key cc@192.5.87.92`   
the `passwd.key` file is located in the folder `pytorch/`.  
    
  
# Install DGL from source
cd  
git clone --recurse-submodules https://github.com/dmlc/dgl.git  
cd dgl/  
<!-- du -h      
#### it's about 1.1GB    -->


#### CUDA build  
mkdir build  
cd build  
cmake -DUSE_CUDA=ON ..  
make -j120  

cd ../python  
sudo python3 setup.py install  
python setup.py build_ext --inplace  

alias python='python3'  
source ~/.bashrc  

Then use 10 files in https://github.com/HaibaraAiChan/dgl_range_sampler.git to replace the 
10 files in the original dgl    
~~~

`Dgl/include/dgl/random.h​`  
   &nbsp; `def RangeInt(lower, upper)​`

`Dgl/include/dgl/aten/csr.h​`   
   &nbsp; CSRRowWiseSampling( range= true)​  

`Dgl/include/dgl/sampling/neighbor.h​`    
    SampleNeighbors_YSY​  

`Dgl/src/array/array_op.h​​`   
   CSRRowWiseSamplingUniformYSY​  

`Dgl/src/array/cuda/rowwise_sampling.cu​`   
   CSRRowWiseSamplingUniformYSY​

`Dgl/python/dgl/sampling/neighbor.py​`   
    def sample_neighbors_range()​    
        DGLGraph.sample_neighbors_range = utils.alias_func(sample_neighbors_range)​

`Dgl/src/graph/sampling/neighbor/neighbor.cc​`   
     DGL_REGISTER_GLOBAL("sampling.neighbor._CAPI_DGLSampleNeighbors_Range")​

`Dgl/src/array/cpu/rowwise_sampling.cc​`   
     GetSamplingYSYRangePickFn()​

`Dgl/src/array/array.cc​`     
    CSRRowWiseSamplingYSY​      

​'Dgl/src/random/cpu/choice.cc'    ​
   YSYChoice​
~~~
Then :    

cd dgl/   
sudo rm -r build/   
mkdir build   
cd build   
cmake -DUSE_CUDA=ON ..  
make -j120   

cd ../python   
sudo python3 setup.py install   







<!-- Buffalo provides bucket-level partitioning and scheduling algorithm.    -->
 
<!-- The overall time complexity of Buffalo’s algorithm (algorithm 3 in the paper)can be summarized as follows:  

### Overall Complexity  
The algorithm's time complexity is:  
**$$O(D + K_{max} \cdot (S + G + M))$$**  

### Components:  
- **$$D$$**: Time for degree bucketing, calculated as **$$O(V + E)$$** (where $$V$$ is nodes and $$E$$ is edges).  
- **$$K_{max}$$**: Maximum number of partitions (micro-batches).  
- **$$S$$**: Time for splitting buckets, **$$O(b)$$**. $$b$$ is the number of output nodes in the bucket to be split.  
- **$$G$$**: Time for balancing memory, calculated as **$$O(n \cdot W)$$** (where $$n$$ is buckets and $$W$$ is memory capacity).  
- **$$M$$**: Time for generating micro-batches, which includes:  
  -  **Parallel Processing**: Can reduce time to **$$O(d)$$** if operations are parallelized. $$d$$ is the degree of center nodes.  



  -->
