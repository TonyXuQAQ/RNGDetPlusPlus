import pickle
import numpy as np
from PIL import Image, ImageDraw
import os
import json

DRIFT = 24
os.makedirs('./vis',exist_ok=True)
with open('./metrics/data_split.json','r') as jf:
    tile_list = json.load(jf)['test'][20:]

def within_margin(v):
    if v[0]>24 and v[0]<400-24 and v[1]>24 and v[1]<400-24:
        return True
    return False

for tile_idx in tile_list:
    print(tile_idx)
    # tile_idx = 'AOI_4_Shanghai_1026'
    # sat_graph = f'./data/20cities/region_{tile_idx}_graph_gt.pickle'
    sat_graph = f'./sat2graph_spacenet_output/{tile_idx}_output_graph_0.05_snap_graph.p'
    RNGDet_graph = f'./RNGDet/test/graph/{tile_idx}_crop.p'
    RNGDetPP_graph = f'./data/RGB_1.0_meter/{tile_idx}__gt_graph_dense_spacenet.p'\
    
    sat_sat = Image.open(f'./data/RGB_1.0_meter/{tile_idx}__rgb.png')
    RNGDet_sat = Image.open(f'./data/RGB_1.0_meter/{tile_idx}__rgb.png')
    RNGDetPP_sat = Image.open(f'./data/RGB_1.0_meter/{tile_idx}__rgb.png')

    ##
    pred_graph = pickle.load(open(RNGDet_graph, "rb"), encoding='latin1')
    pred_draw = ImageDraw.Draw(RNGDet_sat)
    for n, v in pred_graph.items():
        for nei in v:
            pred_draw.line([int(DRIFT+n[1]), int(DRIFT+n[0]),int(DRIFT+nei[1]), int(DRIFT+nei[0])],width=4,fill='orange')
    for n, v in pred_graph.items():
        pred_draw.ellipse([int(DRIFT+n[1]-3), int(DRIFT+n[0]-3), int(DRIFT+n[1]+3), int(DRIFT+n[0]+3)],width=4,fill='yellow')
    RNGDet_sat.save(f'./vis/{tile_idx}_pred.png')

    ##
    pred_graph = pickle.load(open(sat_graph, "rb"), encoding='latin1')
    pred_draw = ImageDraw.Draw(sat_sat)
    for n, v in pred_graph.items():
        for nei in v:
            pred_draw.line([int(DRIFT+n[1]), int(DRIFT+n[0]),int(DRIFT+nei[1]), int(DRIFT+nei[0])],width=4,fill='orange')
    for n, v in pred_graph.items():
        pred_draw.ellipse([int(DRIFT+n[1]-3), int(DRIFT+n[0]-3), int(DRIFT+n[1]+3), int(DRIFT+n[0]+3)],width=4,fill='yellow')
    sat_sat.save(f'./vis/{tile_idx}_sat_pred.png')

    ##
    pred_graph = pickle.load(open(RNGDetPP_graph, "rb"), encoding='latin1')
    pred_draw = ImageDraw.Draw(RNGDetPP_sat)
    for n, v in pred_graph.items():
        for nei in v:
            pred_draw.line([int(DRIFT+n[1]), int(DRIFT+n[0]),int(DRIFT+nei[1]), int(DRIFT+nei[0])],width=4,fill='orange')
    for n, v in pred_graph.items():
        pred_draw.ellipse([int(DRIFT+n[1]-3), int(DRIFT+n[0]-3), int(DRIFT+n[1]+3), int(DRIFT+n[0]+3)],width=4,fill='yellow')
    RNGDetPP_sat.save(f'./vis/{tile_idx}_gt.png')

    break
