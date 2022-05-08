# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import copy
from numpy import save
import datasets.transforms as T
import torch
import torchvision.transforms.functional as F

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
import collections

class globalV(): 
    def __init__(self):
        self.global_v = collections.defaultdict(int)

    def set_value(self, key, value):
        self.global_v[key] = value

    def get_value(self, key):
        return self.global_v[key]

global_value = globalV()


def splitList_by_idx(List, idx):
    idx = set(idx)
    return [v for i, v in enumerate(List) if i in idx]

t_randomErasing = T.RandomErasing(times=20, p=1, scale=(0.005, 0.02), value=0)

def train_one_epoch(args, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    for samples, points_supervision, targets, filenames, is_unlabels in metric_logger.log_every(data_loader, print_freq, header):
        # print(points_supervision[0]['labels'])
        samples = samples.to(device)
        points_supervision = [{k: v.to(device) for k, v in t.items()} for t in points_supervision]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        samples_tensors = samples.tensors
        if args.train_with_unlabel_imgs and sum(is_unlabels) >= 1: 
            # print('====> batch: ', len(targets), 'unlabel nums: ', sum(is_unlabels))
            if len(targets)==sum(is_unlabels): continue
            record_idx = []  
            flip_imgs = []   
            flip_points = []
            for idx, (filename, sample, point_sup, is_unlabel) in enumerate(zip(filenames, samples_tensors, points_supervision, is_unlabels)):
                if is_unlabel: 
                    record_idx.append(idx)
                    flip_img = torch.flip(sample, dims=[-1])
                    if args.dataset_file == 'rsna':
                        flip_img = t_randomErasing(flip_img)
                
                    flip_imgs.append(flip_img.unsqueeze(0))
                    point_sup_ = copy.deepcopy(point_sup)
                    point_sup_['points'][:, 0] = 1 - point_sup_['points'][:, 0]
                    eps = 0.05
                    relative = torch.Tensor(point_sup_['points'].size(0), 2).uniform_(eps, eps).to(point_sup_['points'].device)
                    point_sup_['points'] += relative

                    flip_points.append(point_sup_)

            assert len(flip_imgs) == len(flip_points), 'the number of imgs and anns must be the same.'
            flip_imgs = torch.cat(flip_imgs, dim=0)
            samples = torch.cat([samples_tensors, flip_imgs], dim=0)
            points_supervision += flip_points
        
                

        outputs = model(samples, points_supervision)
        # print('=======original outpus \n', outputs)  
        
        if args.train_with_unlabel_imgs and sum(is_unlabels) >= 1:
            outputs_for_ori     = {'pred_boxes': [], 'aux_outputs': []}
            outputs_for_unlabel = {'pred_boxes': [], 'aux_outputs': []}

            pred_boxes, aux_outputs = outputs['pred_boxes'], outputs['aux_outputs']
            N_extra = len(flip_imgs)

            ori_batch_num = len(targets)
            cur_batch_num = len(points_supervision)
            pred_boxes_ori, pred_boxes_unlabel_flip = pred_boxes[:cur_batch_num - N_extra], pred_boxes[cur_batch_num - N_extra:]
            label_idx = list(set(record_idx) ^ set(range(ori_batch_num)))
            targets = splitList_by_idx(targets, label_idx)
            pred_boxes_ori, pred_boxes_unlabel = splitList_by_idx(pred_boxes_ori, label_idx), splitList_by_idx(pred_boxes_ori, record_idx)
            assert len(pred_boxes_unlabel) == len(pred_boxes_unlabel_flip), 'the number of imgs and flip imgs must be the same.'
            outputs_for_ori['pred_boxes'] += pred_boxes_ori
            outputs_for_unlabel['pred_boxes'].append(torch.cat(pred_boxes_unlabel + pred_boxes_unlabel_flip, dim=0))

            for aux_output in aux_outputs:
                aux_output = aux_output['pred_boxes']
                aux_output_ori, aux_output_unlabel_flip = aux_output[:cur_batch_num - N_extra], aux_output[cur_batch_num - N_extra:]
                aux_output_ori, aux_output_unlabel = splitList_by_idx(aux_output_ori, label_idx), splitList_by_idx(aux_output_ori, record_idx)
                outputs_for_ori['aux_outputs'].append({'pred_boxes': aux_output_ori})
                outputs_for_unlabel['aux_outputs'].append({'pred_boxes': [torch.cat(aux_output_unlabel + aux_output_unlabel_flip, dim=0)]})

            outputs = outputs_for_ori
            unlabel_outputs = outputs_for_unlabel


        loss_dict = criterion(outputs, targets)
        if args.train_with_unlabel_imgs and sum(is_unlabels) >= 1:
            loss_dict_unlabel = criterion(unlabel_outputs, targets=None, specifiec_loss='cal_unlabel_consistency')
            loss_dict.update(loss_dict_unlabel)
        
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if
                                    k in weight_dict and len(k.split('_')) == 2}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        # metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled)
        # metric_logger.update(mIoU=loss_dict_reduced['mIoU'])
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



@torch.no_grad()
def evaluate(args, model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()


    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    if args.save_csv:
        import csv
        csv_write = csv.writer(open(args.save_csv, 'w'))
    for samples, points_supervision, targets, filename, _ in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        points_supervision = [{k: v.to(device) for k, v in t.items()} for t in points_supervision]

        
        # with torch.no_grad():
        # print(points_supervision, targets, filename)
        outputs = model(samples, points_supervision)
        loss_dict = criterion(outputs, targets, 'test')
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if
                                    k in weight_dict and len(k.split('_')) == 2}

        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if args.save_evaltxt_forDraw:  # for draw imgs
            '''
            ==================================================
            {1: {'boxes': (tensor([[599.1519, 465.2538, 844.6331, 838.9862],
                    [165.7746, 445.2140, 408.1470, 832.2156]], device='cuda:0'),)}}
            ==================================================
            {2: {'boxes': (tensor([[224.3862, 465.6223, 481.9965, 832.2103],
                    [617.2076, 491.6592, 871.8964, 857.8474]], device='cuda:0'),)}}
            ==================================================
            {4: {'boxes': (tensor([[629.7487, 255.3514, 896.3071, 670.9658]], device='cuda:0'),)}}
            '''

            root_ = '/apdcephfs/private_kimji/PointDETR_save'
            img_id = targets[0]['image_id'].item()
            points = points_supervision[0]['points'].cpu().numpy()
            with open(os.path.join(root_, str(img_id) + '.txt'), 'w') as fw:
                # print(res[img_id]['boxes'][0].cpu().numpy())
                pts = res[img_id]['labels'][0][0].cpu().numpy()
                pts = list(map(str, str(pts)[1:-1].split()))
                for idx, (box_, label_) in enumerate(zip(res[img_id]['boxes'][0].cpu().numpy(), pts)):
                    fw.write(str(box_)[1:-1] + ' ' + str(points[idx])[1:-1] + ' ' + label_ +'\n')


        if args.save_csv:  # save pseudo labels to .cvc
            idx2class = {1: 'Aortic_enlargement', 2: 'Atelectasis', 3: 'Calcification', 4: 'Cardiomegaly', 5: 'Consolidation', 6: 'ILD', 7: 'Infiltration', 8: 'Lung_Opacity', 9: 'Nodule_Mass', 10: 'Other_lesion', 11: 'Pleural_effusion', 12: 'Pleural_thickening', 13: 'Pneumothorax', 14: 'Pulmonary_fibrosis'}
            if args.dataset_file == 'rsna': idx2class = {1: 'pneumonia', }
            if args.dataset_file == 'cxr8': idx2class = {1: 'Aortic_enlargement', 2: 'Cardiomegaly', 3: 'Pulmonary_fibrosis', 4: 'Pleural_thickening', 5: 'Pleural_effusion', 6: 'Lung_Opacity', 7: 'Nodule_Mass', 8: 'Others'}
            img_id = targets[0]['image_id'].item()
            points = points_supervision[0]['points'].cpu().numpy()
            pts = res[img_id]['labels'][0][0].cpu().numpy()
            pts = list(map(str, str(pts)[1:-1].split()))

            for box_, label_ in zip(res[img_id]['boxes'][0].cpu().numpy(), pts):
                writeLine = [str(filename)[2:-3], -1, -1, -1, -1, idx2class[int(eval(label_))]]   # filename:  ('005852.jpg',)
                box = list(map(str, str(box_)[1:-1].split()))
                writeLine[1:-1] = box
                csv_write.writerow(writeLine)


        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
