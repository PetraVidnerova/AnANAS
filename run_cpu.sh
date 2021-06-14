source .bashrc
cd elcom/AnANAS/examples/
conda activate elcom

ID=test_cpu
CUDA_VISIBLE_DEVICES="" python ../ananas/main.py elcom_cpu.yaml --id $ID > $ID.log 2> $ID.err 



# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64 CUDA_VISIBLE_DEVICES=1 python main.py --type conv --nsga 3 --trainset cifar10.train --testset cifar10.test --id cifar_nsga3_3 1>> cifar_nsga3_3.log 2> err_cifar_nsga3_3.log &
