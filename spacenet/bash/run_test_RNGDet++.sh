CUDA_VISIBLE_DEVICES=0 python main_test.py --savedir RNGDet --device cuda:0 --image_size 400\
 --dataroot ./dataset/ --ROI_SIZE 128 --backbone resnet101 --checkpoint_dir RNGDetNet_best.pt\
 --candidate_filter_threshold 30 --logit_threshold 0.65 --extract_candidate_threshold 0.6 --alignment_distance 10 \
  --multi_scale --instance_seg 
 
 