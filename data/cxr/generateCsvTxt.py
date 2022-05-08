# -----------------------------------------------------------------------------------------------------------------
# convert original train.csv of VinBigData CXR to csv
# Note that we change 14cls to 8cls
# original imgs: normal/abn（15000, res=2788×2446）, we only need abnormal imgs for detection（4394）
# 
# input: 1. original csv file: train.csv
# output: 1. use csv2json.py to generate csv, https://github.com/spytensor/prepare_detection_dataset
#         2. img name remapping(.txt), change original name to easy-read type(e.g 000001.jpg)
# -----------------------------------------------------------------------------------------------------------------

import os, cv2
import csv
import random
from collections import defaultdict
from ensemble_boxes import *
import numpy as np
import torch
from tqdm import tqdm

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

raw_csv = './train.csv'
# 0 - Aortic enlargement
# 1 - Atelectasis
# 2 - Calcification
# 3 - Cardiomegaly
# 4 - Consolidation
# 5 - ILD
# 6 - Infiltration
# 7 - Lung Opacity
# 8 - Nodule/Mass
# 9 - Other lesion
# 10 - Pleural effusion
# 11 - Pleural thickening
# 12 - Pneumothorax
# 13 - Pulmonary fibrosis
# 
idx2class = {1: 'Aortic_enlargement', 2: 'Atelectasis', 3: 'Calcification', 4: 'Cardiomegaly', 5: 'Consolidation', 6: 'ILD', 7: 'Infiltration', 8: 'Lung_Opacity', 9: 'Nodule_Mass', 10: 'Other_lesion',
             11: 'Pleural_effusion', 12: 'Pleural_thickening', 13: 'Pneumothorax', 14: 'Pulmonary_fibrosis'}


# combine Other_lesion, Infiltration, Calcification, ILD, Consolidation, Atelectasis,Pneumothorax into 1 class
New_class2idx = {'Aortic_enlargement': 1, 'Cardiomegaly': 2, 'Pulmonary_fibrosis': 3, 'Pleural_thickening': 4, 'Pleural_effusion': 5, 'Lung_Opacity': 6, 'Nodule_Mass': 7, 
                 'Other_lesion': 8, 'Infiltration': 8, 'Calcification':8, 'ILD': 8, 'Consolidation': 8, 'Atelectasis': 8, 'Pneumothorax': 8}
New_idx2class = {1: 'Aortic_enlargement', 2: 'Cardiomegaly', 3: 'Pulmonary_fibrosis', 4: 'Pleural_thickening', 5: 'Pleural_effusion', 6: 'Lung_Opacity', 7: 'Nodule_Mass', 8: 'Others'}



def defaultdict_list():
    return defaultdict(list)

imgs_name_class = defaultdict(defaultdict_list)
# imgs_name_class
# xxx {'R8': [[1292.0, 554.0, 1477.0, 805.0, 0], [1754.0, 666.0, 1859.0, 744.0, 2]], 'R10': [[1748.0, 669.0, 1872.0, 754.0, 8], [1748.0, 669.0, 1872.0, 754.0, 13]], 'R9': [[1758.0, 664.0, 1863.0, 752.0, 8]]}
# 063319de25ce7edb9b1c6b8881290140 {'R10': [False], 'R8': [False], 'R9': [False]})

raw_csv_lines = csv.reader(open(raw_csv, 'r'))
for line in raw_csv_lines:
    if line[0] == 'image_id': continue # first line
    img_name, class_id, rad_id = line[0], int(line[2]), line[3]
    box = False
    if class_id != 14: 
        new_id = New_class2idx[idx2class[class_id + 1]]
        box = list(map(float, line[4:8])) + [new_id] # xxyy + cls
    imgs_name_class[img_name][rad_id].append(box)

notThreeAnns = list()
all_imgs = list()
normals_imgs = list()
for idx, (k, v) in enumerate(imgs_name_class.items()):
    if len(v)!=3: notThreeAnns.append(k)
    for _, v_ in v.items():
        if not all(v_):
            normals_imgs.append(k)
            break
    all_imgs.append(k)
print('not 3 anns: ', len(notThreeAnns)) #
print('all imgs: ', len(all_imgs))
print('normal imgs: ', len(normals_imgs))



## new csv and remapping file
row_imgname_to_00000_jpgname = './ClsAll8_row_imgname_to_00000_jpgname.txt'
det_csv = './ClsAll8_detetionWBF.csv'
abnormal_imgs = list(set(all_imgs)^set(normals_imgs))
random.shuffle(abnormal_imgs)

imgH, imgW = 3500, 3500 # 2788, 2446
cnt = 0
with open(row_imgname_to_00000_jpgname, 'w') as txtW:
    with open(det_csv, 'w') as csvW:
        csv_write = csv.writer(csvW)
        for k, v in tqdm(imgs_name_class.items()):
            if k not in abnormal_imgs: continue
            jpgname = str(cnt).zfill(6) + '.jpg'
            txt_line = k + ' ' + jpgname + '\n'
            cnt += 1
            txtW.write(txt_line)

            # # WBF here
            boxes_list, labels_list, scores_list = [], [], []
            for _, ann in v.items():
                box_list, label_list, score_list = [], [], []
                for box in ann:
                    x1, y1, x2, y2, label = box[0]/imgW, box[1]/imgH, box[2]/imgW, box[3]/imgH, box[4]
                    box_list.append([x1, y1, x2, y2])
                    label_list.append(int(label))
                    score_list.append(1.0)
                boxes_list.append(box_list)
                labels_list.append(label_list)
                scores_list.append(score_list)
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr=0.5)
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2, label = box[0]*imgW, box[1]*imgH, box[2]*imgW, box[3]*imgH, New_idx2class[int(label)]
                csv_line = [jpgname] + list(map(str, [x1, y1, x2, y2])) + [label]
                csv_write.writerow(csv_line)

            # use one docker ann
            # box_list, label_list, score_list = [], [], []
            # for _, ann in v.items():
            #     box_list, label_list, score_list = [], [], []
            #     for box in ann:
            #         x1, y1, x2, y2, label = box[0]/imgW, box[1]/imgH, box[2]/imgW, box[3]/imgH, box[4]
            #         skipFlag = False
            #         for b in box_list:
            #             if [x1, y1, x2, y2] == b:
            #                 skipFlag = True
            #                 break
            #         if skipFlag: continue # some doctors give the same box different anns.....
            #         box_list.append([x1, y1, x2, y2])
            #         label_list.append(int(label))
            #         score_list.append(1.0)
            #     continue # 
            # for box, label in zip(box_list, label_list):
            #     x1, y1, x2, y2, label = box[0]*imgW, box[1]*imgH, box[2]*imgW, box[3]*imgH, New_idx2class[int(label)]
            #     csv_line = [jpgname] + list(map(str, [x1, y1, x2, y2])) + [label]
            #     csv_write.writerow(csv_line)

print('done.')
print('save detection csv file: ', os.path.join(os.getcwd() + det_csv))
print('save rename txt file: ', os.path.join(os.getcwd() + row_imgname_to_00000_jpgname))


