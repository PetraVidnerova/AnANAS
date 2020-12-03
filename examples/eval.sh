export CUDA_VISIBLE_DEVICES=0
I=0
python evaluate_result.py list-front checkpoint_nsga_haklnv_conv_$I.pkl > eval_$I.log
python evaluate_result.py --conv True eval-front mnist2d.train mnist2d.test checkpoint_nsga_haklnv_conv_$I.pkl 2>eval_$I.err 1>>eval_$I.log

