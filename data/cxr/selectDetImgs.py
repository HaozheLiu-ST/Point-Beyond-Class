import pydicom as dicom
import os
import cv2
import dicom2jpg
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

folder_path = 'xxx/VinBigDatatrain/train/'
jpg_folder_path = '/YOURPATH/data/CXR/VinBigDataTrain_jpg/'
print(folder_path)
with open('row_imgname_to_00000_jpgname.txt', 'r') as fr:
    lines = fr.readlines()
raw2jpg = {}
for line in lines:
    raw, jpg = line.strip().split(' ')
    raw2jpg[raw] = jpg

images_path = os.listdir(folder_path)
for n, image in enumerate(images_path):
    ds = dicom.read_file(os.path.join(folder_path, image))
    image = image.split('.')[0]
    if image not in raw2jpg: continue # 

    data = apply_voi_lut(ds.pixel_array, ds)
    # data = ds.pixel_array
    
    if ds.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(jpg_folder_path, raw2jpg[image]), data)
    if n % 50 == 0:
        print('{} image converted'.format(n))
