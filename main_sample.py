import os
import shutil
from PIL import Image, ImageDraw
import numpy as np
import torch
from tqdm import tqdm 
import argparse
from sampler import Sampler
import json

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

def BC_sample(args):
    
    # ===================
    sample_tag = 'GT' if args.noise==0 else f'noise_{args.noise}' 
    create_directory(f'./{args.savedir}/samples_vis/{sample_tag}',delete=True)
    create_directory(f'./{args.savedir}/samples/{sample_tag}',delete=True)

    # iterate training data
    with open('./dataset/data_split.json','r') as jf:
        tile_list = json.load(jf)['train']

    for i, tile_name in enumerate(tqdm(tile_list)):
        sampler = Sampler(args,tile_name)
        while 1:
            if sampler.finish_current_image:
                break
            # crop
            v_current = sampler.current_coord.copy()
            sat_ROI, label_masks_ROI ,historical_ROI = sampler.crop_ROI(sampler.current_coord)
            # vertices in the next step
            v_nexts, ahead_segments = sampler.step_expert_BC_sampler()
            # save training sample
            gt_probs, gt_coords, list_len = sampler.calcualte_label(v_current,v_nexts)
            np.savez(os.path.join(f'./{args.savedir}/samples/{sample_tag}',f'{tile_name}_{sampler.step_counter}.npz'),sat=sat_ROI.astype(np.uint8),label_masks=label_masks_ROI.astype(np.uint8),\
                historical_ROI=historical_ROI.astype(np.uint8),gt_probs=gt_probs,gt_coords=gt_coords,list_len=list_len,ahead_segments=ahead_segments)
            # visualization
            if sampler.step_counter%1000==0:
                dst = Image.new('RGB',(args.ROI_SIZE*3+5,args.ROI_SIZE*2+5))
                crop_sat_mask = Image.fromarray((sat_ROI).astype(np.uint8))
                crop_history = Image.fromarray((historical_ROI).astype(np.uint8))
                crop_binary_mask = Image.fromarray((label_masks_ROI[:,:,0]).astype(np.uint8))
                crop_keypoint_mask = Image.fromarray((label_masks_ROI[:,:,1]).astype(np.uint8))
                ahead_segment_map = np.zeros((label_masks_ROI.shape[0],label_masks_ROI.shape[1]))
                for segment in ahead_segments:
                    for v in segment:
                        ahead_segment_map[v[1],v[0]] = 255
                dst.paste(crop_sat_mask,(0,0))
                dst.paste(crop_history,(args.ROI_SIZE,0))
                dst.paste(crop_binary_mask,(0,args.ROI_SIZE))
                dst.paste(crop_keypoint_mask,(args.ROI_SIZE,args.ROI_SIZE))
                dst.paste(Image.fromarray(ahead_segment_map.astype(np.uint8)),(args.ROI_SIZE*2,0))
                draw = ImageDraw.Draw(dst)
                for ii in range(2):
                    for jj in range(2):
                        if not (ii==2 and jj==1):
                            delta_x = ii*args.ROI_SIZE+args.ROI_SIZE//2
                            delta_y = jj*args.ROI_SIZE+args.ROI_SIZE//2
                            draw.ellipse([delta_x-1,delta_y-1,delta_x+1,delta_y+1],fill='orange')
                            if list_len: 
                                for kk in range(list_len):
                                    v_next = gt_coords[kk]
                                    draw.ellipse([delta_x-1+(v_next[0]*args.ROI_SIZE//2),delta_y-1+(v_next[1]*args.ROI_SIZE//2),\
                                        delta_x+1+(v_next[0]*args.ROI_SIZE//2),delta_y+1+(v_next[1]*args.ROI_SIZE//2)],fill='cyan')
                
                dst.convert('RGB').save(f'./{args.savedir}/samples_vis/{sample_tag}/{i}_{tile_name}_{sampler.step_counter}.png')
            if sampler.step_counter > args.max_num_frame:
                break
        Image.fromarray(sampler.historical_map.astype(np.uint8)).save(f'./{args.savedir}/samples_vis/{sample_tag}/{i}_{tile_name}_historical_map.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str)
    parser.add_argument('--dataroot', type=str)
    parser.add_argument("--ROI_SIZE", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=4096)
    parser.add_argument("--edge_move_ahead_length", type=int, default=20)
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--noise", type=int)
    parser.add_argument("--max_num_frame", type=int, default=5000)

    args = parser.parse_args()
    BC_sample(args)
