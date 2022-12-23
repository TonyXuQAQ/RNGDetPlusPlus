import os
import numpy as np
import argparse
from PIL import Image, ImageDraw
import shutil
import torch
from torch import nn
import json
from agent import Agent

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

        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1

def valid(args, RNGDetNet):
    # ==============
    RNGDetNet.cuda()
    RNGDetNet.eval()

    # ============== 
    args.agent_savedir = f'{args.savedir}/valid'
    create_directory(f'./{args.savedir}/valid/graph',delete=True)
    create_directory(f'./{args.savedir}/valid/segmentation',delete=True)
    create_directory(f'./{args.savedir}/valid/skeleton',delete=True)
    create_directory(f'./{args.savedir}/valid/vis',delete=True)
    create_directory(f'./{args.savedir}/valid/score',delete=True)
    create_directory(f'./{args.savedir}/valid/json',delete=True)

    # =====================================
    sigmoid = nn.Sigmoid()
    
    # tile list
    with open('./dataset/data_split.json','r') as jf:
        tile_list = json.load(jf)['validation'][:5]

    for i, tile_name in enumerate(tile_list):
        print('====================================================')
        print(f'{i}/{len(tile_list)}: Start processing {tile_name}')
        # initialize an agent
        print(f'STEP 1: Initialize agent and extract candidate initial vertices...')
        agent = Agent(args,RNGDetNet,tile_name)
        print(f'STEP 2: Interative graph detection...')
        while not agent.finish_current_image:
            agent.step_counter += 1
            # crop ROI
            sat_ROI, historical_ROI = agent.crop_ROI(agent.current_coord)
            sat_ROI = torch.FloatTensor(sat_ROI).permute(2,0,1).unsqueeze(0).cuda() / 255.0
            historical_ROI = torch.FloatTensor(historical_ROI).unsqueeze(0).unsqueeze(0).cuda() / 255.0
            # predict vertices in the next step
            outputs = RNGDetNet(sat_ROI,historical_ROI)
            pred_coords = outputs['pred_boxes']
            pred_probs = outputs['pred_logits']
            # agent moves
            # alignment vertices
            alignment_vertices = [[v[0]-agent.current_coord[0]+agent.crop_size//2,
                v[1]-agent.current_coord[1]+agent.crop_size//2] for v in agent.historical_vertices]
            pred_coords_ROI = agent.step(pred_probs,pred_coords,thr=args.logit_threshold)
            
            if agent.step_counter%1==0:
                if agent.step_counter%1000==0:
                    print(f'Iteration {agent.step_counter}...')
                    Image.fromarray(agent.historical_map.astype(np.uint8)).convert('RGB').save(f'./{args.savedir}/valid/graph/{tile_name}_{agent.step_counter}.png')
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
                dst.convert('RGB').save(f'./{args.savedir}/valid/vis/{tile_name}_{agent.step_counter}.png')
                
            # stop action
            if agent.finish_current_image or agent.step_counter>10000:
                print(f'STEP 3: Finsh exploration. Save visualization and graph...')
                # save historical map
                Image.fromarray(agent.historical_map.astype(np.uint8)).convert('RGB').save(f'./{args.savedir}/valid/skeleton/{tile_name}.png')
                # save generated graph
                try:
                    with open(f'./{args.savedir}/valid/json/{tile_name}.json','w') as jf:
                        json.dump(agent.historical_edges,jf)
                except Exception as e:
                    print('Error...')
                    print(e)
   
                break
    
    