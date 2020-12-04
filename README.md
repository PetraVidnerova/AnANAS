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
1. Run evolution using `main.py`, produces a `.pkl` file with the checkpoint (after each iteration)
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
 
## Acknowledgement: 
This work  was partially supported by the TAČR grant TN01111124 
and institutional support of the Institute of Computer Science RVO 67985807.
Thanks to Štěpán Procházka for the idea of multi-model implementation. 
