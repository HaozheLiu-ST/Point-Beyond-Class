import os
import json
import numpy as np
import pandas as pd
import glob
import cv2
import os
import shutil
import random
from sklearn.model_selection import train_test_split
from PIL import Image

seed = 42
np.random.seed(seed)
random.seed(seed)

# ----------------------------------------------------------------------
# func: target .csv -> trainable .coco json
# config: modify 'partial' in line163 to sepcify the box proportion
# input: imgs（line161）and csv file（line162）
# ouput: save annotation file(.json) to saved_coco_path（line164）
# ----------------------------------------------------------------------



#0-bg 8cls(last 7 -> Others)
classname_to_id = {'Aortic_enlargement': 1, 'Cardiomegaly': 2, 'Pulmonary_fibrosis': 3, 'Pleural_thickening': 4, 'Pleural_effusion': 5, 'Lung_Opacity': 6, 'Nodule_Mass': 7, 'Others': 8}


with open('./imgWH.txt', 'r')as fr:
    lines = fr.readlines()
imgWH = {}
for line in lines:
    imgName, WH = line.strip().split(' ')
    imgWH[imgName] = WH # '1.jpg': '1024_1024'

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
        json.dump(instance, open(save_path, 'w'), ensure_ascii=False, indent=2)  

    # txt->COCO
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

    # build cls
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # build img
    def _image(self, path):
        image = {}
        print(path)

        # img = cv2.imread(self.image_dir + path)
        # image['height'] = img.shape[0]
        # image['width'] = img.shape[1]

        W, H = list(map(int, imgWH[imgName].split('_')))
        image['height'] = H
        image['width'] = W

        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # build ann
    def _annotation(self, shape,label):
        # label = shape[-1]
        points = shape[:4]
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = self._get_seg(points)
        annotation['bbox'] = self._get_box(points)
        annotation['point'] = self._get_point(points)
        annotation['iscrowd'] = 0
        annotation['area'] = self._get_area(points)
        return annotation

    # sample pts
    def _get_point(self, points):
        xmin = int(points[0])
        ymin = int(points[1])
        xmax = int(points[2])
        ymax = int(points[3])
        o_width = abs(xmax - xmin)
        o_height = abs(ymax - ymin)
        
        # points = [random.uniform(xmin, xmax), random.uniform(ymin, ymax)]  # whole box
        # points = [random.uniform(xmin+o_width/4, xmax-o_width/4), random.uniform(ymin+o_height/4, ymax-o_height/4)]  # 1/2box
        points = [random.uniform(xmin+o_width/6, xmax-o_width/6), random.uniform(ymin+o_height/6, ymax-o_height/6)] # 2/3box
        # points = [random.uniform(xmin+o_width/10, xmax-o_width/10), random.uniform(ymin+o_height/10, ymax-o_height/10)] # 4/5box
        return points

    # [x1,y1,w,h] 
    def _get_box(self, points):
        min_x = points[0]
        min_y = points[1]
        max_x = points[2]
        max_y = points[3]
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    # area
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
   

if __name__ == '__main__':
    # ----------------------------------------------------------
    # train:test=8:2, split different proportion(box vs point) in training set
    # ----------------------------------------------------------
    # csv_file = "./detetion_1rad.csv"
    # csv_file = "./detetionWBF.csv"
    csv_file = "./ClsAll8_detetionWBF.csv"
    image_dir = "/YOURPATH/data/CXR/VinBigDataTrain_jpg/"
    partial = 20  
    saved_coco_path = "/YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/%sp/"%(str(partial))
    
    total_csv_annotations = {}
    annotations = pd.read_csv(csv_file,header=None).values
    for annotation in annotations:
        key = annotation[0].split(os.sep)[-1]
        value = np.array([annotation[1:]])
        if key in total_csv_annotations.keys():
            total_csv_annotations[key] = np.concatenate((total_csv_annotations[key],value),axis=0)
        else:
            total_csv_annotations[key] = value



    total_keys = list(total_csv_annotations.keys())
    total_keys.sort() # sort by id. Note the ids have been shuffled~
    N = len(total_keys)

    # train=80% test=20%
    trainNum = int(N * 0.8)
    train_keys, val_box_keys = total_keys[:trainNum], total_keys[trainNum:]

    # 5/10/20/30/40
    trainBoxNum = int(trainNum * partial / 100)
    train_box_keys, train_point_keys = train_keys[:trainBoxNum], train_keys[trainBoxNum:]

    print("train_box_n:", len(train_box_keys), "train_point_n:", len(train_point_keys), 'val_box_n:', len(val_box_keys))
    print("total imgs: ", len(train_box_keys) + len(train_point_keys) + len(val_box_keys))


    # build dirs
    if not os.path.exists(saved_coco_path):
        os.makedirs(saved_coco_path)
    if not os.path.exists(saved_coco_path):
        os.makedirs(saved_coco_path)
    if not os.path.exists(saved_coco_path):
        os.makedirs(saved_coco_path)
    
    l2c_box_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    train_box_instance = l2c_box_train.to_coco(train_box_keys)
    l2c_box_train.save_coco_json(train_box_instance, saved_coco_path + 'instances_trainBox.json')

    l2c_point_train = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    train_point_instance = l2c_point_train.to_coco(train_point_keys)
    l2c_point_train.save_coco_json(train_point_instance, saved_coco_path + 'instances_trainPoint.json')

    # once time is ok
    l2c_box_val = Csv2CoCo(image_dir=image_dir,total_annos=total_csv_annotations)
    val_box_instance = l2c_box_val.to_coco(val_box_keys)
    l2c_box_val.save_coco_json(val_box_instance, saved_coco_path + '../instances_val.json')
