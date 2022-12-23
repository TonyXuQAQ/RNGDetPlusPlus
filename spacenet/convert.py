import pickle
import numpy as np
from PIL import Image, ImageDraw
import os
import json


def within_margin(v):
    if v[0]>24 and v[0]<400-24 and v[1]>24 and v[1]<400-24:
        return True
    return False

def convert_gt():
    os.makedirs('./vis',exist_ok=True)
    with open('./metrics/data_split.json','r') as jf:
        tile_list = json.load(jf)['test']

    for tile_idx in tile_list:
        print(tile_idx)
        
        gt_graph = f'./data/RGB_1.0_meter/{tile_idx}__gt_graph_dense.p'
        gt_graph = pickle.load(open(gt_graph, "rb"), encoding='latin1')
        new_graph = {}
        for n, v in gt_graph.items():
            if within_margin(n):
                new_graph[(400-n[0],n[1])] = [(400-u[0],u[1]) for u in v if within_margin(u)]
        
        pickle.dump(new_graph,open(f'./data/RGB_1.0_meter/{tile_idx}_gt_graph.p','wb'),protocol=2)

def convert_pred_RNGDet(dir):
    os.makedirs('./vis',exist_ok=True)
    with open('./metrics/data_split.json','r') as jf:
        tile_list = json.load(jf)['test']

    for tile_idx in tile_list:
        print(tile_idx)
        
        gt_graph = f'./{dir}/test/graph/{tile_idx}.p'
        gt_graph = pickle.load(open(gt_graph, "rb"), encoding='latin1')
        new_graph = {}
        for n, v in gt_graph.items():
            if within_margin(n):
                new_graph[(n[0]-24,n[1]-24)] = [(u[0]-24,u[1]-24) for u in v if within_margin(u)]
        
        pickle.dump(new_graph,open(f'./{dir}/test/graph/{tile_idx}_crop.p','wb'),protocol=2)
    

def convert_pred():
    os.makedirs('./vis',exist_ok=True)
    with open('./metrics/data_split.json','r') as jf:
        tile_list = json.load(jf)['test']

    thresholds = [0.001,0.05,0.01,0.1]
    for thr in thresholds:
        for tile_idx in tile_list:
            print(tile_idx)
            
            pred_graph = f'./sat2graph_spacenet_output/{tile_idx}_output_graph_{thr}_snap_graph.p'
            pred_graph = pickle.load(open(pred_graph, "rb"), encoding='latin1')
            new_graph = {}
            for n, v in pred_graph.items():
                new_graph[(n[0],n[1])] = [(u[0],u[1]) for u in v]
            pickle.dump(new_graph,open(f'./sat2graph_spacenet_output/{tile_idx}_pred_{thr}_graph.p','wb'),protocol=2)
        # break
try:
    convert_pred_RNGDet('RNGDet_multi_ins')
except:
    pass

try:
    convert_pred_RNGDet('RNGDet_ins')
except:
    pass

try:
    convert_pred_RNGDet('RNGDet')
except:
    pass
# convert_pred()