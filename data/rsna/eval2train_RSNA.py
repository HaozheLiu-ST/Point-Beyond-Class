# ---------------------------------------------------------------------------------------
# combine exp_xxx.csv(generated by point data) and residual gt box labels
#
# ---------------------------------------------------------------------------------------
import os
import csv
import json
from os import path

from numpy.lib import save
import torchvision
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import cv2
from PIL import Image


# ----------------------------------------------------------------------
# func: pseudo .csv -> trainable .coco json
# config: modify 'partial' in line31 to sepcify the box proportion
# input: imgs（line34）, gt（line33）and csv（line32）
# ouput: save annotation file(.json) to saved_coco_path（line35）
# ----------------------------------------------------------------------


# -------------------------------------- config -----------------------------------------
idx2class = {1: 'pneumonia', }
classname_to_id = {'pneumonia': 1, }

partial = 50
eval_csv_name = 'exp4_stage1_data50p_1pts_Erase20_jit005_unlabelLossL2Loss200_Load2ptsconsPth.csv'
coco_gt = '/YOURPATH/data/RSNA/cocoAnn/100p/instances_trainBox.json'
image_dir = "/YOURPATH/data/RSNA/RSNA_jpg/"
saved_coco_path = "/YOURPATH/data/RSNA/gt_and_pseudo/"    
pseudo_labels = ' /apdcephfs/share_1290796/PointDETR_saveCsv_RSNA/' + eval_csv_name
write_csv = os.path.join('./PseudoAndGtBox__' + eval_csv_name)  
# ---------------------------------------------------------------------------------------

assert partial == int(pseudo_labels.split('data')[-1].split('p')[0]), 'ERROR: wrong file or wrong data proportion.'


coco = COCO(coco_gt)
ids = list(coco.imgs.keys())
total_num = len(ids)
unlabel_start_idx = int(total_num * partial / 100)
print('===> Supr. proportion: %d%%' %(partial))
print('===> total imgs: ', total_num)
print('===> box-anns: ', unlabel_start_idx)
print('===> pseudo-anns: ', total_num - unlabel_start_idx)

## gen gt csv
with open(write_csv, 'w') as fw:
    csv_write = csv.writer(fw)
    for idx in range(unlabel_start_idx):
        anns = coco.loadAnns(coco.getAnnIds(ids[idx]))
        filename = coco.loadImgs(ids[idx])[0]["file_name"]
        for ann in anns: 
            xmin, ymin, w, h = ann['bbox']
            xmin, ymin, xmax, ymax = xmin, ymin, xmin + w, ymin + h
            line = [filename] + list(map(str, [xmin, ymin, xmax, ymax])) + [idx2class[ann['category_id']]]
            csv_write.writerow(line)

## combine pseudo file to gt csv
command = 'cat ' + pseudo_labels + ' >> ' + write_csv
os.system(command)


class Csv2CoCo:
    def __init__(self,image_dir,total_annos):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = total_annos

    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  # indent=2 更加美观显示

    
    def to_coco(self, keys):
        self._init_categories()
        for key in keys:
            self.images.append(self._image(key))
            shapes = self.total_annos[key]
            for shape in shapes:
                bboxi = []
                for cor in shape[:-1]:
                    bboxi.append(int(cor))
                label = shape[-1]
                annotation = self._annotation(bboxi,label)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

  
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    
    def _image(self, path):
        image = {}
        image['height'] = 1024
        image['width'] = 1024

        image['id'] = self.img_id
        image['file_name'] = path
        return image

  
    def _annotation(self, shape,label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]

    def _get_area(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return (max_x - min_x+1) * (max_y - min_y+1)
    # segmentation
    def _get_seg(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        h = max_y - min_y
        w = max_x - min_x
        a = []
        a.append([min_x,min_y, min_x,min_y+0.5*h, min_x,max_y, min_x+0.5*w,max_y, max_x,max_y, max_x,max_y-0.5*h, max_x,min_y, max_x-0.5*w,min_y])
        return a


## csv2coco

csv_file = write_csv  

total_csv_annotations = {}
annotations = pd.read_csv(csv_file,header=None).values
for annotation in annotations:
    key = annotation[0].split(os.sep)[-1]
    value = np.array([annotation[1:]])
    if key in total_csv_annotations.keys():
        total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
    else:
        total_csv_annotations[key] = value

train_box_keys = list(total_csv_annotations.keys())
assert len(train_box_keys) == total_num, 'ERROR：both mean [pseudo + gt] imgs'

l2c_box_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
train_box_instance = l2c_box_train.to_coco(train_box_keys)
save_file = os.path.join(saved_coco_path, 'train_LableBox_PseudoBox__' + eval_csv_name[:-3] + 'json')
l2c_box_train.save_coco_json(train_box_instance, save_file)
print('===> save json: ', save_file)
print('===> remove tmp .csv file: ', write_csv)
os.system('rm -f ' + write_csv)
print('done.')