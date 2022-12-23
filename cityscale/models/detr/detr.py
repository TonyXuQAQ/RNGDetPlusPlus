# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.util import box_ops
from models.util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm,
                           dice_loss, sigmoid_focal_loss)
from .segmentation_raw import DETRsegm as raw_DETRsegm
from .transformer import build_transformer

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, args,backbone, history_backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.args = args
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if not args.multi_scale:
            self.transformer = transformer
            self.input_proj = nn.Conv2d(backbone.num_channels * 2 , hidden_dim, kernel_size=1)
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        else:
            self.transformer = transformer
            self.input_proj1 = nn.Conv2d(backbone.num_channels//4 , hidden_dim, kernel_size=1)
            self.input_proj2 = nn.Conv2d(backbone.num_channels//2 , hidden_dim, kernel_size=1)
            self.input_proj3 = nn.Conv2d(backbone.num_channels , hidden_dim, kernel_size=1)
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        self.backbone = backbone
        self.history_backbone = history_backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).tanh()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, args):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.l1_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([3]))
        self.bce_loss2 = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([6]))
        self.args = args

    def loss_labels(self, pred_logits, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = pred_logits
        # print(src_logits)
        if indices is not None:
            idx = self._get_src_permutation_idx(indices)
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
            target_classes = torch.full(src_logits.shape[:2], 1,
                                        dtype=torch.int64, device=src_logits.device)
            
            target_classes[idx] = target_classes_o
        else:
            target_classes = torch.full(src_logits.shape[:2], 1,
                                        dtype=torch.int64, device=src_logits.device)
        return F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, pred_boxes, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        return F.l1_loss(src_boxes, target_boxes)

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        src_masks = outputs["pred_masks"]
        masks = [t["masks"].unsqueeze(0) for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks_raw = torch.cat(masks,dim=0)
        target_masks_raw = target_masks_raw.to(src_masks)
        # upsample predictions to the target size
        pred_segment_mask = interpolate(src_masks[:, 0:1], size=target_masks_raw.shape[-2:],
                                mode="bilinear", align_corners=False)
        pred_segment_mask = pred_segment_mask.flatten(1)
        target_mask = target_masks_raw[:,0:1].flatten(1)
        target_mask = target_mask.view(pred_segment_mask.shape)

        pred_point_mask = interpolate(src_masks[:, 1:2], size=target_masks_raw.shape[-2:],
                                mode="bilinear", align_corners=False)
        pred_point_mask = pred_point_mask.flatten(1)
        target_point_mask = target_masks_raw[:,1:2].flatten(1)
        target_point_mask = target_point_mask.view(pred_segment_mask.shape)
        
        return self.bce_loss(pred_segment_mask, target_mask) + self.bce_loss(pred_point_mask, target_point_mask)

    def loss_instance_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
        
        src_masks = outputs["pred_instance_masks"]
        src_masks = src_masks[src_idx]

        target_masks = torch.cat([t['instance_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # masks = [t["instance_masks"] for t in targets]
        # # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        # target_masks = target_masks[tgt_idx]

        target_masks = torch.cat([t['instance_masks'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_masks = target_masks.to(src_masks)
        
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        
        return  self.bce_loss(src_masks, target_masks) 


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'instance_masks':self.loss_instance_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = {'loss_ce':0,'loss_bbox':0,'loss_seg':0,'loss_instance_seg':0}
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs['pred_logits'], outputs['pred_boxes'], targets)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        
        losses['loss_ce'] = self.loss_labels(outputs['pred_logits'], targets, indices, num_boxes)
        losses['loss_bbox'] = self.loss_boxes(outputs['pred_boxes'], targets, indices, num_boxes)

        losses['loss_seg'] = self.loss_masks( outputs, targets, indices, num_boxes)
        if self.args.instance_seg:
            losses['loss_instance_seg'] = self.loss_instance_masks( outputs, targets, indices, num_boxes)
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 1

    backbone = build_backbone(args)

    history_backbone = build_backbone(args,history=True)

    transformer = build_transformer(args)

    model = DETR(
        args,
        backbone,
        history_backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if not args.multi_scale:
        model = raw_DETRsegm(model, freeze_detr=(args.frozen_weights is not None))  
    else:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    losses = ['labels', 'boxes','masks']
    if args.instance_seg:
        losses.append('instance_masks')
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}

    

    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, args=args)
    criterion.cuda()
    return model, criterion
