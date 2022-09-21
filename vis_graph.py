import pickle
import numpy as np
from PIL import Image, ImageDraw
import os
import json

os.makedirs('./vis',exist_ok=True)
with open('./dataset/data_split.json','r') as jf:
    tile_list = json.load(jf)['test']

for tile_idx in tile_list:
    print(tile_idx)
    
    gt_graph = f'./data/20cities/region_{tile_idx}_graph_gt.pickle'
    RNGDet_graph = f'./RNGDet/test/graph/{tile_idx}.p'
    RNGDetPP_graph = f'./RNGDet_multi_ins/test/graph/{tile_idx}.p'\
    
    gt_sat = Image.open(f'./data/20cities/region_{tile_idx}_sat.png')
    RNGDet_sat = Image.open(f'./data/20cities/region_{tile_idx}_sat.png')
    RNGDetPP_sat = Image.open(f'./data/20cities/region_{tile_idx}_sat.png')

    ##
    gt_graph = pickle.load(open(gt_graph, "rb"), encoding='latin1')
    gt_draw = ImageDraw.Draw(gt_sat)
    for n, v in gt_graph.items():
        for nei in v:
            gt_draw.line([int(n[1]), int(n[0]),int(nei[1]), int(nei[0])],width=4,fill='cyan')
    gt_sat.save(f'./vis/{tile_idx}_gt.png')

    ##
    pred_graph = pickle.load(open(RNGDet_graph, "rb"), encoding='latin1')
    pred_draw = ImageDraw.Draw(RNGDet_sat)
    for n, v in pred_graph.items():
        for nei in v:
            pred_draw.line([int(n[1]), int(n[0]),int(nei[1]), int(nei[0])],width=4,fill='orange')
    for n, v in pred_graph.items():
        pred_draw.ellipse([int(n[1]-3), int(n[0]-3), int(n[1]+3), int(n[0]+3)],width=4,fill='yellow')
    RNGDet_sat.save(f'./vis/{tile_idx}_RNGDet.png')

    ##
    pred_graph = pickle.load(open(RNGDetPP_graph, "rb"), encoding='latin1')
    pred_draw = ImageDraw.Draw(RNGDetPP_sat)
    for n, v in pred_graph.items():
        for nei in v:
            pred_draw.line([int(n[1]), int(n[0]),int(nei[1]), int(nei[0])],width=4,fill='orange')
    for n, v in pred_graph.items():
        pred_draw.ellipse([int(n[1]-3), int(n[0]-3), int(n[1]+3), int(n[0]+3)],width=4,fill='yellow')
    RNGDetPP_sat.save(f'./vis/{tile_idx}_RNGDet++.png')
