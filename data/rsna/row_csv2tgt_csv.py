import pydicom
import cv2, os, csv
from collections import defaultdict


root = '/YOURPATH/data/rsna/'
file_label = root + 'stage_2_train_labels.csv'
save_path = './detection.csv'

csv_label = csv.reader(open(file_label, 'r'))
writer = csv.writer(open(save_path, 'w'))

rowname_2_full0 = {}
with open('row_imgname_to_00000_jpgname.txt', 'r') as fr:
    lines = fr.readlines()
for line in lines:
    rowname, full0 = line.strip().split(' ')
    rowname_2_full0[rowname] = full0

for line in csv_label:
    name, x, y, w, h, label = line
    print(line)
    if label == '1':
        line = [rowname_2_full0[name], x, y, str(eval(x) + eval(w)), str(eval(y) + eval(h)), 'pneumonia']
        writer.writerow(line)


