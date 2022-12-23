import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch.distributed as dist
import shutil
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch import nn
from scipy.spatial import cKDTree

from dataset import RNGDet_dataset
from models.detr import build_model
from main_val import valid

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)


def train(args):
    # ==============
    if args.multi_GPU:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(f'cuda:{args.local_rank}')
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False 
        train_loader, train_sampler = RNGDet_dataset(args)
        RNGDetNet, criterion = build_model(args)
        RNGDetNet.cuda()
        RNGDetNet = torch.nn.parallel.DistributedDataParallel(RNGDetNet, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
        model_without_ddp = RNGDetNet.module
    else:
        train_loader = RNGDet_dataset(args)
        RNGDetNet, criterion = build_model(args)
        RNGDetNet.cuda()
        model_without_ddp = RNGDetNet
    
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    opt = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    sched = MultiStepLR(opt, [20,30,40], 0.25)
    if args.local_rank==0:
        # ============== 
        if args.multi_scale:
            args.savedir = f'{args.savedir}_multi'
        if args.instance_seg:
            args.savedir = f'{args.savedir}_ins'
        create_directory(f'./{args.savedir}/tensorboard',delete=True)
        create_directory(f'./{args.savedir}/tensorboard_past')
        create_directory(f'./{args.savedir}/train',delete=True)
        create_directory(f'./{args.savedir}/valid',delete=True)
        create_directory(f'./{args.savedir}/checkpoints')
        writer = SummaryWriter(args.savedir+'/tensorboard')

    sigmoid = nn.Sigmoid()
    #=====================================
    RNGDetNet.train()
    best_f1 = 0
    best_f1_last_10_epoch = 0
    for epoch in range(args.nepochs):
        if args.multi_GPU:
            train_sampler.set_epoch(epoch)
        with tqdm(total=len(train_loader), unit='img') as pbar:
            for i, data in enumerate(train_loader):
                sat, historical_map, label_masks, gt_prob, gt_coord, gt_mask, list_len = data
                sat, historical_map, label_masks, gt_prob, gt_coord, gt_mask = \
                    sat.type(torch.FloatTensor).cuda(), \
                    historical_map.type(torch.FloatTensor).cuda(), \
                    label_masks.type(torch.FloatTensor).cuda(),\
                    gt_prob.type(torch.LongTensor).cuda(), \
                    gt_coord.type(torch.FloatTensor).cuda(), \
                    gt_mask.type(torch.FloatTensor).cuda()
                outputs = RNGDetNet(sat,historical_map)
                targets = [{'labels':gt_prob[x,:list_len[x]],'masks':label_masks[x],'boxes':gt_coord[x,:list_len[x]],'instance_masks':gt_mask[x,:list_len[x]]} for x in range(label_masks.shape[0])]
                loss_dict = criterion(outputs, targets)
                if args.instance_seg:
                    loss_ce, loss_coord, loss_seg, loss_instance_seg = loss_dict['loss_ce'], loss_dict['loss_bbox'] * 5, loss_dict['loss_seg'], loss_dict['loss_instance_seg']
                    loss = loss_ce + loss_coord + loss_seg + loss_instance_seg
                else:
                    loss_ce, loss_coord, loss_seg = loss_dict['loss_ce'], loss_dict['loss_bbox'] * 5, loss_dict['loss_seg']
                    loss = loss_ce + loss_coord + loss_seg
                pred_coords = outputs['pred_boxes'][-1]
                pred_probs = outputs['pred_logits'][-1]
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                # ====================== vis
                if args.local_rank == 0:
                    writer.add_scalar('train/loss_ce', loss_ce, i+epoch*len(train_loader))
                    writer.add_scalar('train/loss_coord', loss_coord, i+epoch*len(train_loader))
                    writer.add_scalar('train/loss_seg', loss_seg, i+epoch*len(train_loader))
                    if args.instance_seg:
                        writer.add_scalar('train/loss_instance_seg', loss_instance_seg, i+epoch*len(train_loader))
                    if i%100==0:
                        # vis
                        pred_binary = sigmoid(outputs['pred_masks'][-1,0]) * 255
                        pred_keypoints = sigmoid(outputs['pred_masks'][-1,1]) * 255
                        
                        
                        # vis
                        dst = Image.new('RGB',(args.ROI_SIZE*4+5,args.ROI_SIZE*2+5))
                        sat = Image.fromarray((sat[-1].permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
                        history = Image.fromarray((historical_map[-1,0].cpu().detach().numpy()*255).astype(np.uint8))
                        gt_binary = Image.fromarray((label_masks[-1,0].cpu().detach().numpy()*255).astype(np.uint8))
                        gt_keypoint = Image.fromarray((label_masks[-1,1].cpu().detach().numpy()*255).astype(np.uint8))
                        pred_binary = Image.fromarray((pred_binary.cpu().detach().numpy()).astype(np.uint8))
                        pred_keypoint = Image.fromarray((pred_keypoints.cpu().detach().numpy()).astype(np.uint8))
                        
                        dst.paste(sat,(0,0))
                        dst.paste(history,(0,args.ROI_SIZE))
                        dst.paste(gt_binary,(args.ROI_SIZE,0))
                        dst.paste(gt_keypoint,(args.ROI_SIZE*2,0))
                        dst.paste(pred_binary,(args.ROI_SIZE,args.ROI_SIZE))
                        dst.paste(pred_keypoint,(args.ROI_SIZE*2,args.ROI_SIZE))

                        if args.instance_seg:
                            gt_instance_mask = Image.fromarray(np.clip((torch.sum(gt_mask[-1],dim=0)*255).cpu().detach().numpy(),0,255).astype(np.uint8))
                            dst.paste(gt_instance_mask,(args.ROI_SIZE*3,0))
                            pred_logits = pred_probs.softmax(dim=1)
                            pred_logits = [x.unsqueeze(0) for ii,x in enumerate(outputs['pred_instance_masks'][-1].sigmoid()) if pred_logits[ii][0]>=args.logit_threshold]
                            if len(pred_logits):
                                pred_instance_mask = torch.cat(pred_logits,dim=0)
                                pred_instance_mask = Image.fromarray(np.clip((torch.sum(pred_instance_mask,dim=0)*255).cpu().detach().numpy(),0,255).astype(np.uint8))
                                dst.paste(pred_instance_mask,(args.ROI_SIZE*3,args.ROI_SIZE))                                               
                        draw = ImageDraw.Draw(dst)
                        for ii in range(3):
                            for jj in range(2):
                                if not (ii==2 and jj==1):
                                    delta_x = ii*args.ROI_SIZE+args.ROI_SIZE//2
                                    delta_y = jj*args.ROI_SIZE+args.ROI_SIZE//2
                                    draw.ellipse([delta_x-1,delta_y-1,delta_x+1,delta_y+1],fill='orange')
                                    if list_len[-1]: 
                                        for kk in range(list_len[-1]):
                                            v_next = gt_coord.cpu().detach().numpy()[-1,kk]
                                            draw.ellipse([delta_x-1+(v_next[0]*args.ROI_SIZE//2),delta_y-1+(v_next[1]*args.ROI_SIZE//2),\
                                                delta_x+1+(v_next[0]*args.ROI_SIZE//2),delta_y+1+(v_next[1]*args.ROI_SIZE//2)],fill='cyan')

                                    for jj in range(pred_coords.shape[0]):
                                            v = pred_coords[jj]
                                            v = [delta_x+(v[0]*args.ROI_SIZE//2),delta_y+(v[1]*args.ROI_SIZE//2)]
                                            if pred_probs[jj][0] < pred_probs[jj][1]:
                                                draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='yellow',outline='yellow')
                                            else:
                                                draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='pink',outline='pink')
                        dst.convert('RGB').save(f'./{args.savedir}/train/{i}.png')
                        

                if args.multi_GPU:
                    dist.barrier()
                if args.instance_seg:
                    pbar.set_description(f'===Epoch: {epoch} | ce/seg/instance/coord: {round(loss_ce.item(),3)}/{round(loss_seg.item(),3)}/{round(loss_instance_seg.item(),3)}/{round(loss_coord.item(),3)} ')
                else:
                    pbar.set_description(f'===Epoch: {epoch} | ce/seg/coord: {round(loss_ce.item(),3)}/{round(loss_seg.item(),3)}/{round(loss_coord.item(),3)} ')
                pbar.update()
                # break
        
        if args.local_rank==0:
            torch.save(model_without_ddp.state_dict(),os.path.join(args.savedir+'/checkpoints', f"RNGDetNet_{epoch}.pt"))
            print(f'Save checkpoint {epoch}')
            print('Start evaluation.....')
            precision, recall, f1 = evaluate(args, model_without_ddp)
            f1 += epoch/1000.0
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model_without_ddp.state_dict(),os.path.join(args.savedir+'/checkpoints', f"RNGDetNet_best.pt"))
            if epoch > args.nepochs - 10 and f1 > best_f1_last_10_epoch:
                best_f1_last_10_epoch = f1
                torch.save(model_without_ddp.state_dict(),os.path.join(args.savedir+'/checkpoints', f"RNGDetNet_best_last_10_epoch.pt"))
            print(f'precision/recall/f1: {precision}/{recall}/{f1}')
            writer.add_scalar('eval/precision', precision, epoch)
            writer.add_scalar('eval/recall', recall, epoch)
            writer.add_scalar('eval/f1', f1, epoch)
        if args.multi_GPU:
            dist.barrier()
        RNGDetNet.train()
        sched.step()

def evaluate(args,RNGDetNet):
    def calculate_scores(gt_points,pred_points):
        if not len(gt_points):
            return 0,0,0
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


    RNGDetNet.eval()
    valid(args,RNGDetNet)
    
    scores = []
    for name in os.listdir(f'./{args.savedir}/valid/skeleton'):
        pred_graph = np.array(Image.open(f'./{args.savedir}/valid/skeleton/{name}'))[args.ROI_SIZE:-args.ROI_SIZE,args.ROI_SIZE:-args.ROI_SIZE]
        gt_graph = np.array(Image.open(f'./dataset/segment/{name}'))
        scores.append(pixel_eval_metric(pred_graph,gt_graph))
    return round(sum([x[0] for x in scores])/(len(scores)+1e-7),3),\
            round(sum([x[1] for x in scores])/(len(scores)+1e-7),3),\
            round(sum([x[2] for x in scores])/(len(scores)+1e-7),3)


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

    parser.add_argument('--multi_GPU',  action='store_true')
    parser.add_argument("--nepochs", type=int, default=1000)
    parser.add_argument("--nworkers", type=int, default=4)
    parser.add_argument("--ROI_SIZE", type=int, default=256)
    parser.add_argument("--orientation_channels", type=int, default=2)
    parser.add_argument("--segmentation_channels", type=int, default=3)
    parser.add_argument("--noise", type=int, default=7)
    
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
    train(args)
