# ------------------------------------------------------------------------
# UP-DETR
# Copyright (c) Tencent, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .rsna import build as build_rsna
from .my_sampler import MyBalancedSampler

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    elif args.dataset_file == 'rsna':
        return build_rsna(image_set, args)
    elif args.dataset_file == 'cxr8':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def build_mySampler(dataset_train, batch_size, partial, args):
    return MyBalancedSampler(dataset_train, batch=batch_size, partial=partial, args=args)
