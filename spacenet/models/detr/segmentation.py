# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file provides the definition of the convolutional heads used to predict masks, as well as the losses
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.util.misc import NestedTensor,  nested_tensor_from_tensor_list
from torch import Tensor
import copy

class DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.point_segmentation_head = FPN()
        self.segment_segmentation_head = FPN()
        self.multi_scale_head = multi_scale(self.detr.args, detr.transformer, hidden_dim, nheads, [2048, 1024, 512, 256], detr.args.ROI_SIZE)
        
    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='bilinear')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y

    def forward(self, samples: NestedTensor, history_samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        out = {}
        features, pos = self.detr.backbone(samples)
        bs = features[-1].tensors.shape[0]
        pred_segment_mask = self.segment_segmentation_head([features[3].tensors,features[2].tensors, features[1].tensors, features[0].tensors])
        pred_keypoint_mask = self.point_segmentation_head([features[3].tensors,features[2].tensors, features[1].tensors, features[0].tensors])
        cat_tensor = torch.cat([pred_segment_mask, pred_keypoint_mask],dim=1)
        out["pred_masks"] = cat_tensor
        segmentation_map = cat_tensor.clone().detach().sigmoid()
        history_samples = torch.cat([history_samples,segmentation_map],dim=1)
        if isinstance(history_samples, (list, torch.Tensor)):
            history_samples = nested_tensor_from_tensor_list(history_samples)

        features2, _ = self.detr.history_backbone(history_samples)
        # ===================== 

        if self.detr.args.instance_seg:
            hs, seg_masks = self.multi_scale_head(features,features2,self.detr.query_embed.weight,pos)
            outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
            out["pred_instance_masks"] = outputs_seg_masks
        else:
            hs = self.multi_scale_head(features,features2,self.detr.query_embed.weight,pos)
        
        out["pred_logits"] = (self.detr.class_embed(hs))
        out["pred_boxes"] = (self.detr.bbox_embed(hs).tanh())

        return out

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

class multi_scale(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, args, transformer, dim, nheads ,fpn_dims, output_size):
        super(multi_scale, self).__init__()
        self.in_planes = 64
        self.args = args
        self.dim = dim
        self.fpn_dims = fpn_dims
        self.output_size = output_size
        # Top layer
        self.transformer = transformer
        
        self.input_proj_layer1 = nn.Conv2d(fpn_dims[0], dim, kernel_size=1)
        self.input_proj_layer2 = nn.Conv2d(fpn_dims[1], dim, kernel_size=1)
        self.input_proj_layer3 = nn.Conv2d(fpn_dims[2], dim, kernel_size=1)
        self.input_proj_layer4 = nn.Conv2d(fpn_dims[3], dim, kernel_size=1)
        #
        self.bbox_attention = MHAttentionMap(dim, dim, nheads, dropout=0.0)

        self.lay1 = torch.nn.Conv2d(fpn_dims[0]+nheads, dim+nheads, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim+nheads)
        self.lay2 = torch.nn.Conv2d(dim+nheads, dim//2+nheads, 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, dim//2+nheads)
        self.lay3 = torch.nn.Conv2d(dim//2+nheads,dim//4+nheads, 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, dim//4+nheads)
        self.lay4 = torch.nn.Conv2d(dim//4+nheads,dim//8+nheads, 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, dim//8+nheads)
        self.lay5 = torch.nn.Conv2d(dim//8+nheads, dim//16+nheads, 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, dim//16+nheads)
        self.out_lay = torch.nn.Conv2d(dim//16+nheads, 1, 3, padding=1)

        self.adapter1 = torch.nn.Conv2d(fpn_dims[1]+nheads, dim//2+nheads, 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[2]+nheads, dim//4+nheads, 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[3]+nheads, dim//8+nheads, 1)

    def _get_clones(self,module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self, features, features2, query_embed_weight, pos):
        c2, c3, c4, c5 = features

        c2 = c2.tensors + features2[0].tensors
        c3 = c3.tensors + features2[1].tensors
        c4 = c4.tensors + features2[2].tensors
        c5 = c5.tensors + features2[3].tensors

        # no fuse
        
        mask = torch.ones((c5.shape[0],c5.shape[2],c5.shape[3]), dtype=torch.bool).cuda()
        mask[:,:] = False
        t5 = self.input_proj_layer1(c5)
        hs4, memory4 = self.transformer(t5, mask, query_embed_weight, pos[-1])
        bbox_mask4 = self.bbox_attention(hs4[-1], memory4)

        # first fuse. (2048,H/32,W/32) + (1024,H/16,W/16) => fpn feature (1024,H/16,W/16) + transformer feature (256,H/16,W/16)
        mask = torch.ones((c4.shape[0],c4.shape[2],c4.shape[3]), dtype=torch.bool).cuda()
        mask[:,:] = False
        t4 = self.input_proj_layer2(c4)
        hs3, memory3 = self.transformer(t4, mask, hs4[-1], pos[-2])
        bbox_mask3 = self.bbox_attention(hs3[-1], memory3)

        # second fuse. (1024,H/16,W/16) + (512,H/8,W/8) => fpn feature (512,H/8,W/8) + transformer feature (256,H/8,W/8)
        mask = torch.ones((c3.shape[0],c3.shape[2],c3.shape[3]), dtype=torch.bool).cuda()
        mask[:,:] = False
        t3 = self.input_proj_layer3(c3)
        hs2, memory2 = self.transformer(t3, mask, hs3[-1], pos[-3])
        bbox_mask2 = self.bbox_attention(hs2[-1], memory2)

        # third fuse. (512,H/8,W/8) + (256,H/4,W/4) => fpn feature (256,H/4,W/4) + transformer feature (256,H/4,W/4)
        mask = torch.ones((c2.shape[0],c2.shape[2],c2.shape[3]), dtype=torch.bool).cuda()
        mask[:,:] = False
        t2 = self.input_proj_layer4(c2)
        hs1, memory1 = self.transformer(t2, mask, hs2[-1], pos[-4])
        bbox_mask1 = self.bbox_attention(hs1[-1], memory1)

        if self.args.instance_seg:
            x = torch.cat([_expand(c5, bbox_mask4.shape[1]), bbox_mask4.flatten(0, 1)], 1)
            x = self.lay1(x)
            x = self.gn1(x)
            x = F.relu(x)
            x = self.lay2(x)
            x = self.gn2(x)
            x = F.relu(x)
            
            cur_fpn = self.adapter1(torch.cat([_expand(c4, bbox_mask3.shape[1]), bbox_mask3.flatten(0, 1)], 1))
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            x = self.lay3(x)
            x = self.gn3(x)
            x = F.relu(x)

            cur_fpn = self.adapter2(torch.cat([_expand(c3, bbox_mask2.shape[1]), bbox_mask2.flatten(0, 1)], 1))
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            x = self.lay4(x)
            x = self.gn4(x)
            x = F.relu(x)

            cur_fpn = self.adapter3(torch.cat([_expand(c2, bbox_mask1.shape[1]), bbox_mask1.flatten(0, 1)], 1))
            x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
            x = self.lay5(x)
            x = self.gn5(x)
            x = F.relu(x)

            x = self.out_lay(x)
            
            x = F.interpolate(x,  size=(self.output_size,self.output_size), mode='bilinear', align_corners=True)
            return hs1[-1], x
        return hs1[-1]

class InstanceSeg(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self):
        super(InstanceSeg, self).__init__()
        self.in_planes = 64
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_fuse = nn.Conv2d(128, 8, kernel_size=7, stride=1, padding=0)
        self.output_layer = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self,fpns):
        c5, c4, c3, c2 = fpns

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))


        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)


        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        feature = self._upsample(self.conv_fuse(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        output = self.output_layer(feature)
        return output

class FPN(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self):
        super(FPN, self).__init__()
        self.in_planes = 64
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

		# Semantic branch
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_fuse = nn.Conv2d(128, 8, kernel_size=7, stride=1, padding=0)
        self.output_layer = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # num_groups, num_channels
        self.gn1 = nn.GroupNorm(128, 128) 
        self.gn2 = nn.GroupNorm(256, 256)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y


    def forward(self,fpns):
        c5, c4, c3, c2 = fpns

        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))


        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)


        # Semantic
        _, _, h, w = p2.size()
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # 256->128
        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)

        # 256->256
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)

        # 256->128
        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)

        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        feature = self._upsample(self.conv_fuse(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        output = self.output_layer(feature)
        return output




def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.75, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

