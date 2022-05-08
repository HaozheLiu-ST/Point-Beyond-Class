# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import sys
import numpy as np
import cv2
import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import datasets.transforms as T

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, args, image_set, img_folder, ann_file, transforms, return_masks, is_training, sample_points_num):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.args = args
        self.image_set = image_set

        if args.generate_pseudo_bbox: 
            assert not is_training and args.partial != 0, ' ERROR: The data scale must be specified.'
            if args.dataset_file == 'coco':
                self.coco = COCO('/YOURPATH/data/CXR/cocoAnnWBF_2-3box/100p/instances_trainBox.json')
            elif args.dataset_file == 'cxr8':
                self.coco = COCO('/YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/100p/instances_trainBox.json')
            else:
                pass
            self.ids = list(self.coco.imgs.keys())
            N = int(len(self.ids) * args.partial / 100)
            self.ids = self.ids[N:]
            print('\n ====> The amount of data that needs to generate pseudo-labels: %d, (training data=%d)' %(len(self.ids), N), '\n')

        else: 
            # point data
            if args.train_with_unlabel_imgs and image_set == 'train':
                print('=====> training with unlabeled imgs (point annotation)')
                if args.dataset_file == 'coco':
                    all_box_anns = '/YOURPATH/data/CXR/cocoAnnWBF_2-3box/100p/instances_trainBox.json'
                elif args.dataset_file == 'cxr8':
                    all_box_anns = '/YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/100p/instances_trainBox.json'
                else:
                    pass
                self.all_anns = COCO(all_box_anns)
                print('=====>', all_box_anns)
                self.ids_all_anns = list(set(list(sorted(self.all_anns.imgs.keys()))))
                self.ids_unlabel_anns = list(set(self.ids_all_anns) ^ set(self.ids))
                self.ids = self.ids_all_anns
                self.coco = self.all_anns


        self.is_training = is_training
        self.sample_points_num = sample_points_num
        print('sample points num .. ', self.sample_points_num)

    def __getitem__(self, idx):
        image_id = self.ids[idx]

        img, target = super(CocoDetection, self).__getitem__(idx)
        filename = self.coco.loadImgs(self.ids[idx])[0]["file_name"]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        if self.is_training:
            if self.args.train_with_unlabel_imgs and idx in self.ids_unlabel_anns:
                points_supervision, target = generate_target_evaluation(target) 
            else:
                points_supervision, target = generate_target_training(target, self.sample_points_num)

            # from engine import global_value
            # points_supervision, target = generate_target_training(target, global_value.get_value())
        else:
            points_supervision, target = generate_target_evaluation(target)

        if self.args.train_with_unlabel_imgs and self.image_set == 'train' :
            is_unlabel = idx in self.ids_unlabel_anns
            if is_unlabel: del target['boxes']
            return img, points_supervision, target, filename, is_unlabel
        else:
            return img, points_supervision, target, filename, None

def generate_target_training(target, K=1):
    boxes = target['boxes']
    labels = target['labels']
    N = len(boxes)

    object_ids = torch.arange(N)

    eps = 0.01

    relative_x = torch.Tensor(N, K).uniform_(-1/3 + eps, 1/3 - eps)  # 2/3 box
    relative_y = torch.Tensor(N, K).uniform_(-1/3 + eps, 1/3 - eps)

    x = boxes[:, 0, None] + boxes[:, 2, None] * relative_x
    y = boxes[:, 1, None] + boxes[:, 3, None] * relative_y
    x, y = x.reshape(-1), y.reshape(-1)


    points = torch.stack([
        x, y
    ], dim=1)

    boxes = boxes[:, None, :].repeat(1, K, 1).flatten(0, 1)
    labels = labels[:, None].repeat(1, K).reshape(-1)
    object_ids = object_ids[:, None].repeat(1, K).reshape(-1)

    return {'labels': labels, 'object_ids': object_ids, 'points': points}, {'boxes': boxes}

def generate_target_evaluation(target):
    K = 1
    boxes = target['boxes']
    labels = target['labels']
    N = len(boxes)

    object_ids = torch.arange(N)
    points = target['points'][:, :]


    boxes = boxes[:, None, :].repeat(1, K, 1).flatten(0, 1)
    labels = labels[:, None].repeat(1, K).reshape(-1)
    object_ids = object_ids[:, None].repeat(1, K).reshape(-1)

    return {'labels': labels, 'object_ids': object_ids, 'points': points}, {'boxes': boxes, 'orig_size': target['orig_size'], 'image_id': target['image_id'], 'anno_ids': target['anno_ids']}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        points = [obj["point"] for obj in anno]
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 2) # x,y
        points[:, 0].clamp_(min=0, max=w) #..  x, y  or  y, x??
        points[:, 1].clamp_(min=0, max=h)

        ids = [obj["id"] for obj in anno]
        ids = torch.tensor(ids, dtype=torch.int64)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        points = points[keep]
        classes = classes[keep]
        ids = ids[keep]

        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["points"] = points
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["anno_ids"] = ids
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, data_augment=False, is_training=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if image_set == 'train':
        if is_training:
            if data_augment:
                scales = [360, 480, 512, 576, 608]
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomSelect(
                        T.RandomResize([512], max_size=512),
                        T.Compose([
                            T.RandomResize([512, 600, 700]),
                            T.RandomSizeCrop(512, 512),
                        ])
                    ),
                    normalize,
                ])
            else:
                return T.Compose([
                    T.RandomResize([512], max_size=512),
                    normalize,
                ])

    return T.Compose([
        T.RandomResize([512], max_size=512),
        normalize,
    ])


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    PATHS = {
        "train": ('/YOURPATH/data/CXR/VinBigDataTrain_jpg', root/ 'instances_trainBox.json'),
        "val": ('/YOURPATH/data/CXR/VinBigDataTrain_jpg', root/ '../instances_val.json'),
    }
    img_folder, ann_file = PATHS[image_set]
    print('dataset path: \n', ann_file)
    dataset = CocoDetection(
        args,
        image_set, 
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args.data_augment, not (args.eval or args.generate_pseudo_bbox)),
        return_masks=args.masks,
        is_training=not (args.eval or args.generate_pseudo_bbox or image_set == "val"),
        sample_points_num=args.sample_points_num
    )
    return dataset
