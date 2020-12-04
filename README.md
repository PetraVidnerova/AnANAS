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

numpy, keras, pandas, scikit-learn, deap 

## Usage:
```
usage: main.py [-h] [--type TYPE] [--trainset TRAINSET] [--testset TESTSET]
               [--nsga NSGA] [--id ID] [--checkpoint CHECKPOINT]
               [--config CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  --type TYPE           either "conv" or "dense"
  --trainset TRAINSET   filename of training set
  --testset TESTSET     filename of test set
  --nsga NSGA           0,1,2,3
  --id ID               computation id
  --checkpoint CHECKPOINT
                        checkpoint file to load the initial state from
  --config CONFIG       json config filename

```

Example:
```
python main.py --type dense --trainset DATASET_NAME --testset DATASET_NAME --id EXPERIMENT_ID
```
to run on one GPU (recommended), I use: 
```
CUDA_VISIBLE_DEVICES=0 python main.py --type dense --trainset DATASET_NAME --testset DATASET_NAME --id EXPERIMENT_ID
```

 
## Acknowledgement: 
This work  was partially supported by the TAČR grant TN01111124 
and institutional support of the Institute of Computer Science RVO 67985807.
Thanks to Štěpán Procházka for the idea of multi-model implementation. 
