import os
import numpy as np
import argparse
from PIL import Image, ImageDraw
import shutil
import torch
from torch import nn
import json
from models.detr import build_model
from agent import Agent
import time
from scipy.spatial import cKDTree
import pickle

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

class Vertex():
    def __init__(self,v,id):
        self.x = v[0]
        self.y = v[1]
        self.id = id
        self.neighbors = []

class Edge():
    def __init__(self,src,dst,id):
        self.src = src
        self.dst = dst
        self.id = id

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertex_num = 0
        self.edge_num = 0

    def find_v(self,v_coord):
        if f'{v_coord[0]}_{v_coord[1]}' in self.vertices.keys():
            return self.vertices[f'{v_coord[0]}_{v_coord[1]}']
        return 

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.find_v(v1_coord)
        if v1 is None:
            v1 = Vertex(v1_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v1.x}_{v1.y}'] = v1
        
        v2 = self.find_v(v2_coord)
        if v2 is None:
            v2 = Vertex(v2_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v2.x}_{v2.y}'] = v2

        if v1 not in v2.neighbors:
            v2.neighbors.append(v1)
        if v2 not in v1.neighbors:
            v1.neighbors.append(v2)
        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1

def calculate_scores(gt_points,pred_points):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0,0,0
    thr = 3
    dis_gt2pred,_ = pred_tree.query(gt_points, k=1)
    dis_pred2gt,_ = gt_tree.query(pred_points, k=1)
    recall = len([x for x in dis_gt2pred if x<thr])/len(dis_gt2pred)
    acc = len([x for x in dis_pred2gt if x<thr])/len(dis_pred2gt)
    r_f = 0
    if acc*recall:
        r_f = 2*recall * acc / (acc+recall)
    return acc, recall, r_f

def pixel_eval_metric(pred_mask,gt_mask):
    def tuple2list(t):
        return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

    gt_points = tuple2list(np.where(gt_mask!=0))
    pred_points = tuple2list(np.where(pred_mask!=0))

    return calculate_scores(gt_points,pred_points)

def test(args):
    # ============== 
    if args.multi_scale:
        args.savedir = f'{args.savedir}_multi'
    if args.instance_seg:
        args.savedir = f'{args.savedir}_ins'

    # ==============
    RNGDetNet, criterion = build_model(args)
    torch.cuda.set_device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False 
    RNGDetNet.load_state_dict(torch.load(f'{args.savedir}/checkpoints/{args.checkpoint_dir}',map_location='cpu'))
    RNGDetNet.cuda()

    args.agent_savedir = f'./{args.savedir}/test'

    print('=======================')
    print(f'candidate_filter_threshold: {args.candidate_filter_threshold}')
    print(f'logit_threshold: {args.logit_threshold}')
    print(f'extract_candidate_threshold: {args.extract_candidate_threshold}')
    print(f'save directory: {args.savedir}')
    print('=======================')
    
    create_directory(f'./{args.savedir}/test/graph',delete=True)
    create_directory(f'./{args.savedir}/test/segmentation',delete=True)
    create_directory(f'./{args.savedir}/test/skeleton',delete=True)
    create_directory(f'./{args.savedir}/test/vis',delete=True)
    create_directory(f'./{args.savedir}/test/score',delete=True)
    create_directory(f'./{args.savedir}/test/json',delete=True)
    create_directory(f'./{args.savedir}/test/results/apls',delete=True)
    create_directory(f'./{args.savedir}/test/results/topo',delete=True)

    # =====================================
    RNGDetNet.eval()
    sigmoid = nn.Sigmoid()
    
    # tile list
    with open('./dataset/data_split.json','r') as jf:
        tile_list = json.load(jf)['test']
    time_start = time.time()
    for i, tile_name in enumerate(tile_list):
        print('====================================================')
        print(f'{i}/{len(tile_list)}: Start processing {tile_name}')
        # initialize an agent
        print(f'STEP 1: Initialize agent and extract candidate initial vertices...')
        time1 = time.time()
        agent = Agent(args,RNGDetNet,tile_name)
        print(f'STEP 2: Interative graph detection...')
        while 1:
            agent.step_counter += 1
            # crop ROI
            sat_ROI, historical_ROI = agent.crop_ROI(agent.current_coord)
            sat_ROI = torch.FloatTensor(sat_ROI).permute(2,0,1).unsqueeze(0).cuda() / 255.0
            historical_ROI = torch.FloatTensor(historical_ROI).unsqueeze(0).unsqueeze(0).cuda() / 255.0
            # predict vertices in the next step
            outputs = RNGDetNet(sat_ROI,historical_ROI)
            # agent moves
            # alignment vertices
            pred_coords = outputs['pred_boxes']
            pred_probs = outputs['pred_logits']
            alignment_vertices = [[v[0]-agent.current_coord[0]+agent.crop_size//2,
                v[1]-agent.current_coord[1]+agent.crop_size//2] for v in agent.historical_vertices]
            pred_coords_ROI = agent.step(pred_probs,pred_coords,thr=args.logit_threshold)
            
            if agent.step_counter%100==0:
                if agent.step_counter%1000==0:
                    print(f'Iteration {agent.step_counter}...')
                    Image.fromarray(agent.historical_map[args.ROI_SIZE:-args.ROI_SIZE,args.ROI_SIZE:-args.ROI_SIZE].astype(np.uint8)).convert('RGB').save(f'./{args.agent_savedir}/skeleton/{tile_name}_{agent.step_counter}.png')
                    # vis
                pred_binary = sigmoid(outputs['pred_masks'][0,0]) * 255
                pred_keypoints = sigmoid(outputs['pred_masks'][0,1]) * 255
                # vis
                dst = Image.new('RGB',(args.ROI_SIZE*3+5,args.ROI_SIZE*2+5))
                sat = Image.fromarray((sat_ROI[0].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
                history = Image.fromarray((historical_ROI[0,0].cpu().detach().numpy()*255).astype(np.uint8))
                pred_binary = Image.fromarray((pred_binary.cpu().detach().numpy()).astype(np.uint8))
                pred_keypoint = Image.fromarray((pred_keypoints.cpu().detach().numpy()).astype(np.uint8))
                dst.paste(sat,(0,0))
                dst.paste(history,(0,args.ROI_SIZE))
                dst.paste(pred_binary,(args.ROI_SIZE,0))
                dst.paste(pred_keypoint,(args.ROI_SIZE,args.ROI_SIZE))
                if args.instance_seg:
                    pred_logits = pred_probs[-1].softmax(dim=1)
                    pred_logits = [x.unsqueeze(0) for ii,x in enumerate(outputs['pred_instance_masks'][-1].sigmoid()) if pred_logits[ii][0]>=args.logit_threshold]
                    if len(pred_logits):
                        pred_instance_mask = torch.cat(pred_logits,dim=0)
                        pred_instance_mask = Image.fromarray(np.clip((torch.sum(pred_instance_mask,dim=0)*255).cpu().detach().numpy(),0,255).astype(np.uint8))
                        dst.paste(pred_instance_mask,(args.ROI_SIZE*2,0))
                draw = ImageDraw.Draw(dst)
                for ii in range(3):
                    for kk in range(2):
                        delta_x = ii*args.ROI_SIZE
                        delta_y = kk*args.ROI_SIZE
                        if len(alignment_vertices):
                            for v in alignment_vertices:
                                if v[0]>=0 and v[0]<agent.crop_size and v[1]>=0 and v[1]<agent.crop_size:
                                    v = [delta_x+(v[0]),delta_y+(v[1])]
                                    draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='cyan',outline='cyan')

                        if pred_coords_ROI:
                            for jj in range(len(pred_coords_ROI)):
                                v = pred_coords_ROI[jj]
                                v = [delta_x+(v[0]),delta_y+(v[1])]
                                draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='pink',outline='pink')
                        
                        draw.ellipse([delta_x-1+args.ROI_SIZE//2,delta_y-1+args.ROI_SIZE//2,delta_x+1+args.ROI_SIZE//2,delta_y+1+args.ROI_SIZE//2],fill='orange')
                dst.convert('RGB').save(f'./{args.agent_savedir}/vis/{tile_name}_{agent.step_counter}.png')
                
            # stop action
            if agent.finish_current_image:
                print(f'STEP 3: Finsh exploration. Save visualization and graph...')
                # save historical map
                Image.fromarray(agent.historical_map[args.ROI_SIZE:-args.ROI_SIZE,args.ROI_SIZE:-args.ROI_SIZE].astype(np.uint8)).convert('RGB').save(f'./{args.savedir}/test/skeleton/{tile_name}.png')
                # save generated graph
                graph = Graph()
                try:
                    with open(f'./{args.savedir}/test/json/{tile_name}.json','w') as jf:
                        json.dump(agent.historical_edges,jf)
                except Exception as e:
                    print('Error...')
                    print(e)

                for e in agent.historical_edges:
                    graph.add(e)
                
                output_graph = {}

                for k, v in graph.vertices.items():
                    output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

                pickle.dump(output_graph,open(f'./{args.savedir}/test/graph/{tile_name}.p','wb'),protocol=2)

                pred_graph = np.array(Image.open(f'./{args.savedir}/test/skeleton/{tile_name}.png'))
                gt_graph = np.array(Image.open(f'./data/segment/{tile_name}.png'))
                print(pixel_eval_metric(pred_graph,gt_graph))
                break
        time2 = time.time()
        print(f'{i}/{len(tile_list)}: Finish processing {tile_name}, time usage {round((time2-time1)/3600,3)}h')
            
    time_end = time.time()
    print(f'Finish inference, time usage {round((time_end-time_start)/3600,3)}h')     
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HDMapNet training.')
    # logging config
    parser.add_argument("--savedir", type=str)

    # nuScenes config
    parser.add_argument('--dataroot', type=str)

    # loss config
    parser.add_argument("--scale_seg", type=float, default=1.0)
    parser.add_argument("--scale_var", type=float, default=1.0)
    parser.add_argument("--scale_dist", type=float, default=1.0)
    parser.add_argument("--scale_direction", type=float, default=0.2)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_channels', default=128+64, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=10, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=False,
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--device',default='cuda:0',type=str)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--checkpoint_dir',  type=str)
    parser.add_argument("--ROI_SIZE", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=4096)
    parser.add_argument("--logit_threshold", type=float, default=0.8)
    parser.add_argument("--candidate_filter_threshold", type=int, default=50)
    parser.add_argument("--extract_candidate_threshold", type=float, default=0.65)
    parser.add_argument("--alignment_distance", type=int, default=10)
    parser.add_argument("--filter_distance", type=int, default=10)
    parser.add_argument("--multi_scale", action='store_true')
    parser.add_argument("--instance_seg", action='store_true')
    parser.add_argument("--process_boundary", action='store_true')
    
    args = parser.parse_args()
    test(args)
