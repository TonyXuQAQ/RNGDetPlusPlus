CUDA_VISIBLE_DEVICES=0 python main_test.py --savedir RNGDet --device cuda:0 --image_size 2048\
 --dataroot ./dataset/ --ROI_SIZE 128 --backbone resnet101 --checkpoint_dir RNGDet_best.pt\
 --candidate_filter_threshold 30 --logit_threshold 0.75 --extract_candidate_threshold 0.7 --alignment_distance 5
 
 