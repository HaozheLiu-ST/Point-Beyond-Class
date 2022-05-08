import pydicom
import cv2, os, csv
from collections import defaultdict
import random
import shutil
import numpy as np

seed = 42
np.random.seed(seed)
random.seed(seed)

root = '/YOURPATH/data/rsna/'
file_label = root + 'stage_2_train_labels.csv'
save_path = '/YOURPATH/data/RSNA/RSNA_jpg'
one2one_file = 'row_imgname_to_00000_jpgname.txt'

csv_label = csv.reader(open(file_label))
dic = defaultdict(list)
for line in csv_label:
    name, x, y, w, h, label = line
    dic[name].append([x,y,w,h,label])

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    print(len(dict_key_ls))
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

dic = random_dic(dic)

with open(one2one_file, 'w') as fw:
    index = 0
    for name, value in dic.items():
        if value[0][-1] != '1': continue
        
        imgPath = os.path.join(root, 'stage_2_train_images/' + name + '.dcm')
        ds = pydicom.read_file(imgPath)
        img = ds.pixel_array

        jpg_name = str(index).zfill(6) + '.jpg'
        cv2.imwrite(os.path.join(save_path, jpg_name), img)
        print(name, jpg_name)
        fw.write(name + ' ' + jpg_name + '\n')
        index += 1


