# export CUDA_VISIBLE_DEVICES=0 
# python main.py --type conv --trainset mnist2d.train --testset mnist2d.test --id haklnv_conv 1> test_haklnv_conv.log 2> err_haklnv_conv.log 

# for I in `seq 0 4`
# do
#     CUDA_VISIBLE_DEVICES=$I python main.py --type conv --trainset mnist2d.train --testset mnist2d.test --id icaisc_$I 1> icaisc_$I.log 2> err_icaisc_$I.log &
# done

I=1 
python main.py --type conv --trainset fashion_mnist.train --testset fashion_mnist.test --id icaisc_fashion_$I 1> icaisc_fashion_$I.log 2> err_icaisc_fashion_$I.log &

