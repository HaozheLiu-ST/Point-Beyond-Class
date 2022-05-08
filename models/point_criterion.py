from typing import Sized
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)



class PointCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, weight_dict, eos_coef, losses, args):
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
        # self.matcher = matcher
        self.weight_dict = weight_dict
        self.args = args

        """
        weight_dict .. 
        {
            'loss_ce'  : 1, 'loss_bbox'  : 5, 'loss_giou'  : 2, 
            'loss_ce_0': 1, 'loss_bbox_0': 5, 'loss_giou_0': 2, 
            'loss_ce_1': 1, 'loss_bbox_1': 5, 'loss_giou_1': 2, 
            'loss_ce_2': 1, 'loss_bbox_2': 5, 'loss_giou_2': 2, 
            'loss_ce_3': 1, 'loss_bbox_3': 5, 'loss_giou_3': 2, 
            'loss_ce_4': 1, 'loss_bbox_4': 5, 'loss_giou_4': 2
        }
        """

        print('weight dict .. ', weight_dict)
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)


    def loss_labels(self, outputs, targets, num_boxes, log=True):
        raise NotImplementedError('loss labels is not implemented... ')


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, num_boxes):
        raise NotImplementedError('loss labels is not implemented... ')


    def loss_consistency(self, outputs, targets, num_boxes):
        src_boxes = torch.cat(outputs['pred_boxes'], dim=0) # N * 4
        loss_cons = torch.mean(torch.pow(src_boxes[1::2,:] - src_boxes[0::2,:], 2))
        losses = {'loss_cons': loss_cons}
        return losses

    def loss_unlabelconsistency(self, outputs, targets, num_boxes):
        src_boxes = torch.cat(outputs['pred_boxes'], dim=0) # N * 4  xywh
        N = src_boxes.size(0)
        assert N % 2 == 0, 'must be even'
        N = N // 2
        src_boxes[N:, 0] = 1.0 - src_boxes[N:, 0] # xywh -> (1-x)ywh
        loss_unlabelcons = torch.pow(src_boxes[N:,:] - src_boxes[:N,:], 2).sum() / N  # L2
        # loss_unlabelcons = (F.l1_loss(src_boxes[:N,:], src_boxes[N:,:], reduction='none')).sum() / N  # L1

        losses = {'loss_unlabelcons': loss_unlabelcons}
        return losses
        

    def loss_boxes(self, outputs, targets, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # idx = self._get_src_permutation_idx(indices)
        # print(outputs['pred_boxes'])
        src_boxes = torch.cat(outputs['pred_boxes'], dim=0)
        
        # src_boxes = outputs['pred_boxes'][idx]

        target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)

        # target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        #debug
        box_ = box_ops.box_cxcywh_to_xyxy(src_boxes)
        if not (box_[:, 2:] >= box_[:, :2]).all():
            print('bug here \n')
            print('src_box', src_boxes)
            print('box xyxy:', box_)
            return losses

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    """
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
    """

    def get_loss(self, loss, outputs, targets, num_boxes, **kwargs):
        loss_map = {
            # 'labels': self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            # 'masks': self.loss_masks
            'consistency': self.loss_consistency,
            'unlabel_consistency': self.loss_unlabelconsistency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, num_boxes, **kwargs)

    def forward(self, outputs, targets, phase='train', specifiec_loss=None):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        if not specifiec_loss:
            num_boxes = sum(len(t["boxes"]) for t in targets)
            num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=targets[0]["boxes"].device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                if phase=='test' and loss=='consistency': continue
                losses.update(self.get_loss(loss, outputs, targets, num_boxes))

            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    # indices = self.matcher(aux_outputs, targets)
                    for loss in self.losses:
                        if phase=='test' and loss=='consistency': continue
                        if loss == 'masks':
                            # Intermediate masks losses are too costly to compute, we ignore them.
                            continue
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss(loss, aux_outputs, targets, num_boxes, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

            return losses

        elif specifiec_loss == 'cal_unlabel_consistency':
            losses = {}
            losses.update(self.get_loss('unlabel_consistency', outputs, targets, num_boxes=0))
            
            # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
            if 'aux_outputs' in outputs:
                for i, aux_outputs in enumerate(outputs['aux_outputs']):
                    for loss in self.losses:
                        kwargs = {}
                        if loss == 'labels':
                            # Logging is enabled only for the last layer
                            kwargs = {'log': False}
                        l_dict = self.get_loss('unlabel_consistency', aux_outputs, targets, num_boxes=0, **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
            return losses