# <img src="fig/ananas.png" width="50px"/> AnANAS
Answer for Automatic Neural Architecture Search 


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
python main.py --type dense --trainset DATASET_NAME --testset DATASET_NAME --id ID_TEXT
```
to run on one GPU (recommended), I use: 
```
CUDA_VISIBLE_DEVICES=0 python main.py --type dense --trainset DATASET_NAME --testset DATASET_NAME --id ID_TEXT
```

 
## Acknowledgement: 
This work  was partially supported by the TAČR grant TN01111124 
and institutional support of the Institute of Computer Science RVO 67985807.
