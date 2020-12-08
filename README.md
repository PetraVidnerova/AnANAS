# <img src="fig/ananas.png" width="50px"/> AnANAS
Answer to Automatic Neural Architecture Search 


*Author: Petra Vidnerová, The Czech Academy of Sciences, Institute of Computer Science*

A tool for automatic architecture search based on multiobjective evolution optimising 
both network perfomance and an architecture size.  

## Keywords:
- automatic model selection
- neural architecture search 
- multiobjective optimisation
- NSGA, NSGAII, NSGA3
- deep neural networks 
- convolutional neural networks

## Requirements:

numpy, keras, pandas, scikit-learn, deap, click, matplotlib

## Main features:
- using keras datasets or data form csv files 
- "vanilla" GA, multiobjective evolution via NSGA, NSGAII, NSGAIII 
- runing in parallel on one GPU or parallel on several CPUs 
- optimising feedworfard deep neural networks with dense layers, convolutinal networks   
 
## Usage:
1. Run evolution using `main.py`, produces a `.json` file with the list of all architectures from the pareto-front, a `.pkl` file with the checkpoint (after each iteration). Checkpoint stores all information
 needed to continue the computation in another run as well as the results. 
2. Inspect results runing `evaluate_result.py` on the resulting `.pkl` checkpoint file 

### main.py: 
```
usage: main.py [-h] [--id ID] [--log LOG] TASK

positional arguments:
  TASK        name of a yaml file with a task definition

optional arguments:
  -h, --help  show this help message and exit
  --id ID     computation id
  --log LOG   set logging level, default INFO
```

Example:
```
python main.py task.yaml --id test1
```
to run on one GPU (recommended), I use: 
```
CUDA_VISIBLE_DEVICES=0 python main.py tasks.yaml --id gpu_test 
```
### evaluate_result.py 
```
Usage: evaluate_result.py [OPTIONS] COMMAND [ARGS]...

Options:
  --conv BOOLEAN
  --help          Show this message and exit.

Commands:
  eval-front
  evaluate
  list-front
  plot
  query-iter
``` 

Example:
```
python evaluate_result.py eval-front  --data_source keras --trainset mnist checkpoint.pkl
```

## Config file example

task.yaml 
```
dataset:
  name: mnist
  source_type: keras

network_type: dense

nsga: 2

main_alg:
  batch_size: 128
  eval_batch_size: 30
  epochs: 10
  loss: mean_squared_error
  task_type: classification
  final_epochs: 20

ga:
  pop_size: 20
  n_gen: 20
  
network:
  max_layers: 5
  max_layer_size: 1000
  min_layer_size: 10
  dropout: [0.0, 0.2, 0.3, 0.4]
  activations: ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

device:
  device_type: CPU
  n_cpus: 10 
```

To run on GPU specify:
```
device:
  device_type: GPU
``` 
In fact, the usage of GPU or CPU depands on your tensorflow version. With tensorflow-gpu it will run on GPU. Option `device_type: GPU` 
only forces the  fitness function to use a multi-model and evaluate individuals simutaneously on one GPU.
If no GPU, use `device_type: CPU` and `n_cpus` that forces a use of multiprocessing with the given number of workers. 

To evolve convolutional networks:
```
network_type: conv
``` 
To use data from `csv` file (file should be comma separated, 
without a header, output variable in the last column):
```
dataset:
  source_type: csv
  name: data_train.csv
  test_name: data_test.csv
```

 
## Acknowledgement: 
This work  was partially supported by the TAČR grant TN01111124 
and institutional support of the Institute of Computer Science RVO 67985807.
Thanks to Štěpán Procházka for the idea of multi-model implementation. 
