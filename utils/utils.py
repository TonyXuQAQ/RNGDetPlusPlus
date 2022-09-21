#!/user/bin/python
# coding=utf-8

import os
import torch
from lib import geom, graph as graph_helper
import numpy as np
import logging
import time
import cv2 as cv


def load_pretrained(model, fname, optimizer=None, strict=True):
    """
    resume training from previous checkpoint
    :param fname: filename(with path) of checkpoint file
    :return: model, optimizer, checkpoint epoch
    """
    if os.path.isfile(fname):
        print("=> loading checkpoint '{}'".format(fname))
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['state_dict'], strict=strict)
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            return model, optimizer
        else:
            return model
    else:
        print("=> no checkpoint found at '{}'".format(fname))


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def numpy2tensor2cuda(batch_inputs):
    return torch.autograd.Variable(torch.from_numpy(batch_inputs).float()).cuda()


def get_logger(logger_name="logtrain", log_dir="data/logs/"):
    timenow = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

    logging_dir = os.path.join(log_dir)
    if not os.path.isdir(logging_dir):
        os.makedirs(logging_dir, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt))
    logger.addHandler(console)

    logname = logger_name + "_" + timenow + '.txt'
    handler = logging.FileHandler(os.path.join(logging_dir, logname))
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    return logger


class MapContainer(object):
    def __init__(self, path, region_name, IMG_SZ):
        self.map = np.zeros((2, IMG_SZ, IMG_SZ))
        self.path = path
        self.region_name = region_name

    def add_map(self, pnt, map, CROP_SZ):
        self.map[0, pnt[0]:pnt[0] + CROP_SZ, pnt[1]:pnt[1] + CROP_SZ] += map
        self.map[1, pnt[0]:pnt[0] + CROP_SZ, pnt[1]:pnt[1] + CROP_SZ] += 1

    def add_batch_gpu(self, pnt_lst, maps_cuda, CROP_SZ):
        maps_np = torch.sigmoid(maps_cuda).data.cpu().numpy()
        for batch_i, pnt in enumerate(pnt_lst):
            self.add_map(pnt, maps_np[batch_i, 0, :, :], CROP_SZ)

    def add_batch_cpu(self, pnt_lst, maps_np, CROP_SZ):
        for batch_i, pnt in enumerate(pnt_lst):
            self.add_map(pnt, maps_np[batch_i, 0, :, :], CROP_SZ)

    def close(self):
        self.map[0] /= self.map[1]

    def save_map(self):
        cv.imwrite(os.path.join(self.path, self.region_name + ".png"), self.map[0].swapaxes(0, 1) * 255)

    def get_map(self):
        return self.map[0]

def random_sample_given_probs(seq, probs):
    sum_probs = sum(probs)
    if sum_probs != 1:
        probs = [x/sum_probs for x in probs]
    probs.insert(0, 0)
    for i in range(len(probs)-1):
        probs[i+1] = probs[i] + probs[i+1]
    rand = random.random()
    for i in range(len(probs)-1):
        if probs[i] < rand < probs[i+1]:
            break
    return seq[i]