
# HVPSL

Official Code for the following article: 

> **Xiaoyuan Zhang, Xi Lin, Bo Xue, Yifan Chen, Qingfu Zhang. Hypervolume Maximization: A Geometric View of Pareto Set Learning. NeurIPS 2023** <br/>

To begin with, please install the following packages:

``` 
    conda env create -f env.yml
    conda activate hvpsl
```

If creating from env.yml fail, you can simply pip install the following packages. 

``` 
    conda install numpy 
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install scipy
    pip install pymoo
    pip install tqdm
    pip install cvxpy
    pip install cvxopt
    pip install matplotlib
```


## Run the code
python psl.py --problem-name zdt1 

## To run all methods
run the run_main.sh file. 

