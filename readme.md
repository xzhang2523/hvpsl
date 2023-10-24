
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
    conda create --name hvpsl
    conda activate hvpsl

    conda install numpy 
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install scipy
    pip install pymoo
    pip install tqdm
    pip install cvxpy
    pip install cvxopt
    pip install matplotlib
    pip install nni
    pip install csv2latex
```


## Run the code
Please use the ''decompose'' argument to specify the decomposition method. 
psl-hv1 : hv1. psl-hv2 : hv2. 
```
python psl.py --problem-name zdt1 --decompose hv
```

## To run all methods
run the run_main.sh file. 

## To use NNI to select the reference point
```
nnictl create --config config.yml --port 8080
```

