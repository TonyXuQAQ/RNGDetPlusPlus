CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 main_train.py --savedir RNGDet\
 --dataroot ./dataset/ --batch_size 20 --ROI_SIZE 128 --nepochs 50 --multi_GPU --backbone resnet101 --eos_coef 0.2\
 --lr 1e-4 --lr_backbone 1e-4 --weight_decay 1e-5 --noise 8 --image_size 2048\
  --candidate_filter_threshold 30 --logit_threshold 0.75 --extract_candidate_threshold 0.55 --alignment_distance 5