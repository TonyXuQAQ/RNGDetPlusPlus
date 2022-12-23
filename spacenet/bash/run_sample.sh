python main_sample.py --savedir data --dataroot ./dataset/ --image_size 400 --ROI_SIZE 128\
    --edge_move_ahead_length 30 --num_queries 10 --noise 8 --max_num_frame 10000

python main_sample.py --savedir data --dataroot ./dataset/ --image_size 400 --ROI_SIZE 128\
    --edge_move_ahead_length 30 --num_queries 10 --noise 0 --max_num_frame 10000 --empty_segmentation 
