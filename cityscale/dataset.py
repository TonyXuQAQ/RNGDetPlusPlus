import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import cv2

class RNGDetNet(Dataset):
    def __init__(self,args,is_train=True):
        self.image_list = []
        self.args = args  
        self.is_train = is_train

        data_list = []

        image_path = f'./data/samples/noise_{args.noise}'
        data_list = os.listdir(image_path)
        self.data_list = [os.path.join(image_path,x) for x in data_list]

        # image_path = f'./RNGDet/samples/GT'
        # data_list = os.listdir(image_path)
        # self.data_list += [os.path.join(image_path,x) for x in data_list]
        random.shuffle(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self,idx):
        data = self.data_list[idx]
        sample = np.load(data,allow_pickle=True)
        # random augmentation
        sat = sample['sat'] / 255
        sat[:,:,0] = sat[:,:,0] * (0.7 + 0.3 * random.random())  
        sat[:,:,1] = sat[:,:,1] * (0.7 + 0.3 * random.random())  
        sat[:,:,2] = sat[:,:,2] * (0.7 + 0.3 * random.random()) 
        rot_index = np.random.randint(0,4)

        gt_masks = np.zeros((self.args.ROI_SIZE,self.args.ROI_SIZE,self.args.num_queries))
        kernel = np.ones((4,4), np.uint8)
        
        for ii,segment in enumerate(sample['ahead_segments']):
            for v in segment:
                try:
                    gt_masks[v[1],v[0],ii] = 255
                except:
                    print(segment)
                    raise Exception
            gt_masks[:,:,ii] = cv2.dilate(gt_masks[:,:,ii], kernel, iterations=1)

        theta = rot_index * np.pi / 2
        R = np.array([[np.cos(theta),np.sin(theta)],[np.sin(-theta),np.cos(theta)]])
        gt_coords = R.dot(sample['gt_coords'].T).T
        label_masks = np.rot90(sample['label_masks'],rot_index,[0,1]).copy()
        historical_map = np.rot90(sample['historical_ROI'],rot_index,[0,1]).copy()
        sat = np.rot90(sat,rot_index,[0,1]).copy()
        gt_masks = np.rot90(gt_masks,rot_index,[0,1]).copy()

        sat, historical_map, label_masks, gt_probs, gt_coords, gt_masks, list_len = \
            torch.FloatTensor(sat).permute(2,0,1),\
            torch.FloatTensor(historical_map).unsqueeze(0)/255,\
            torch.FloatTensor(label_masks).permute(2,0,1)/255,\
            torch.LongTensor(sample['gt_probs']).reshape(self.args.num_queries),\
            torch.FloatTensor(gt_coords).reshape(self.args.num_queries,2),\
            torch.FloatTensor(gt_masks).permute(2,0,1)/255, sample['list_len']
        return sat, historical_map, label_masks, gt_probs, gt_coords, gt_masks, list_len


def RNGDet_dataset(args):
    
    train_dataset = RNGDetNet(args, is_train=True)
    if args.multi_GPU:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.nworkers,pin_memory=True,sampler=train_sampler)
        print(f'Training data: {len(train_dataset)}')
        return train_dataloader, train_sampler
    else:
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nworkers)
    print(f'Training data: {len(train_dataset)}')
    return train_dataloader
