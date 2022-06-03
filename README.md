# Point-Beyond-Class

> The official Pytorch Implementation for the paper, 'Point Beyond Class: A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays' (Accepted by MICCAI'2022)



## Step0, convert the original data to standard COCO format

* Two datasets

    Download RSNA and VinDr-CXR datasets from Kaggle, the original directory is as follows
    ```
    |—— rsna
    |    |—— stage_2_train_labels.csv  # annotation file
    |    └── stage_2_train_images      # 26684 imgs（.dcm）
    └── VinBigData Chest X-ray
         |—— train.csv  # annotation file
         └── train      # 15000 imgs(.dicom)
    ```  

* convert RSNA to COCO format
    ```
    cd data/rsna
    python rowimg2jpg.py  # .dim -> .jpg
    python row_csv2tgt_csv.py  # original .csv -> target .csv
    python csv2coco_rsna.py  # check the instructions in this script, you should run multiple times to generate the corresponding JSON file.
    ```
    After that, the generated JSON files can be used directly for training and testing. For example, the path of JSON file can be:
    ```
    /YOURPATH/data/RSNA/cocoAnn/*p/*.json
    ```


* convert VinDr-CXR to COCO format
    ```
    cd data/cxr
    python selectDetImgs.py # .dim -> .jpg
    python generateCsvTxt.py # original .csv -> target .csv & an TXT file for recording the img name remapping.
    python csv2coco_CXR8.py # check the instructions in this script, you should run multiple times to generate the corresponding JSON file.
    ```
    After that, the generated JSON files can be used directly for training and testing. For example, the path of JSON file can be:
    ```
    /YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/*p/*.json
    ```

## Step1-1, training teacher model (beseline, point->box) using partial box annotations of training set.
* Sample code is as follows, see line6-19 in `start_rsna.sh` and line6-19 in `start_CXr8.sh` for details. Note the for CXR dataset, the input points are randomly sampled in the middle 2/3 side length area of the box.
    ```
    partial=20
    python main.py \
    --epochs 111 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --pre_norm \
    --coco_path xxx/$[partial]p \
    --dataset_file rsna or cxr8 \
    --batch_size 16 \
    --num_workers 16 \
    --data_augment \
    --position_embedding v4 \
    --output_dir xxx
    ```
    where, 
    `--dataset_file` specifies dataset; 
    `--coco_path` specifies the path of annotation files

## Step1-2, training teacher model (add Symmetric Consistency, point->box) using all box annotations of training set.
* Sample code is as follows, see line22-39 in `start_rsna.sh` and line41-58 in `start_CXr8.sh` for details. Note the for CXR dataset, the input points are randomly sampled in the middle 2/3 side length area of the box.
    ```
    partial=20
    python main.py \
    --epochs 111 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --pre_norm \
    --coco_path xxx/$[partial]p \
    --dataset_file rsna or cxr8 \
    --batch_size 16 \
    --num_workers 16 \
    --data_augment \
    --position_embedding v4 \
    --sample_points_num 1  \
    --train_with_unlabel_imgs \
    --unlabel_cons_loss_coef 50 \
    --partial $partial \
    --output_dir xxx
    ```
    where,  
    `--train_with_unlabel_imgs` makes sure to use points data;  
    `--unlabel_cons_loss_coef` specifies the weight of Symmetric Consistency loss


## Step1-3, training teacher model (add Multi-Point Consistency, point->box) using partial box annotations of training set.
* Sample code is as follows, see line41-48 in `start_rsna.sh` and line22-37 in `start_CXr8.sh` for details. Note the for CXR dataset, the input points are randomly sampled in the middle 2/3 side length area of the box.
    ```
    partial=20
    python main.py \
    --epochs 111 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --pre_norm \
    --coco_path xxx/$[partial]p \
    --dataset_file rsna or cxr8 \
    --batch_size 16 \
    --num_workers 16 \
    --data_augment \
    --position_embedding v4 \
    --sample_points_num 2 \
    --cons_loss \
    --cons_loss_coef 100 \
    --output_dir xxx
    ```
    Among them，  
    `--sample_points_num` specifies the number of sampled points；  
    `--cons_loss_coef` specifies the weight of Multi-Point Consistency loss  

## Step1-4, training teacher model (add Symmetric Consistency (pre-training) and Multi-Point Consistency, point->box) using all box annotations of training set.
* Sample code is as follows, see line61-79 in `start_rsna.sh` and line61-79 in `start_CXr8.sh` for details. Note the for CXR dataset, the input points are randomly sampled in the middle 2/3 side length area of the box.
    ```
    partial=20
    python main.py \
    --epochs 111 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --pre_norm \
    --coco_path xxx/$[partial]p \
    --dataset_file rsna or cxr8 \
    --batch_size 16 \
    --num_workers 16 \
    --data_augment \
    --position_embedding v4 \
    --sample_points_num 1  \
    --train_with_unlabel_imgs \
    --unlabel_cons_loss_coef 50 \
    --partial $partial \
    --load_from xxx/checkpoint0110.pth \
    --output_dir xxx
    ```
    Among them，  
    `--load_from` specifies the pre-trained models training from step 1-3；  


## Step2, generate pseudo labels offline (point->box, using partial point annotations of training set)
* Sample inference code is as follows, see line82-94 in `start_rsna.sh` and `start_CXr8.sh` for details.
    ```
    python main.py \
    --batch_size 1 \
    --num_workers 8 \
    --eval \
    --no_aux_loss \
    --dataset_file rsna or cxr8 \
    --pre_norm \
    --coco_path Not important, but you need to specify an existing path \
    --position_embedding v4 \
    --resume xxx/checkpoint0110.pth \
    --save_csv xxx.csv \
    --generate_pseudo_bbox \
    --partial 20
    ```
    where,  
    `--resume` specifies the teacher model used to generate pseudo labels;  
    `--save_csv` specifies the intermediate file(.csv) to save pseudo labels;
    `--generate_pseudo_bbox` make sure to generate pseudo labels; 
    `--partial` specifies the proportion of point annotaions. 


* Combine GT labels and pseudo labels to generate trainable JSON file(coco format)
    ```
    --- for rsna ---
    cd data/rsna
    python eval2train_RSNA.py  # You need to specify the pseudo-label intermediate file (.csv generated by the command above) and proportion in the script, same as below.
    --- for cxr8 ---
    cd data/cxr
    python eval2train_CXR8.py 
    ```

## Step3, train student model(box -> box, using mmdetection toolkit，version2.8)
* training types 
    a) Only train gt boxes, i.e box annotations in training set
    b) train gt boxes + pseudo boxes， i.e gt box annotations in training set + pseudo box annotations generated from point annotations using point DETR
    c) train gt boxes + pseudo boxes， i.e gt box annotations in training set + pseudo box annotations generated from point annotations using PBC
*  Sample code is as follows, which can be found in `./student/start_xxx.sh`
    ```
    python3 tools/train.py \
        configs/faster/faster_xxx.py \ # or fcos
        --seed 42 \
        --deterministic \
        --work-dir xxx
    ```
    > Please Note that you have to specify the target training annotation file in corresponding config file manually. For example:
    * In `faster_CXR8.py`  
        - specify data proportion __p in line151, to train type a)  
        - specify data proportion __p in line154, to train type b)  
        - specify data proportion __p in line157, to train type c)  
    * `fcos_CXR8.py`  
        - specify data proportion __p in line49, to train type a)  
        - specify data proportion __p in line52, to train type b)  
        - specify data proportion __p in line55, to train type c)  

## Other
* Visualize 'epoch-map' curves from log file 
    ```
    cd pyScripts/
    python drawLogCXR8.py
    python drawLogRSNA.py
    ```
    Note that all the training logs are available in `./outfiles/logs`

* Pre-trained models
Pre-trained models can be found [here](https://drive.google.com/drive/folders/1TzGoaFDs36OQ9W5xLLeQZJ1s_oFaiihV?usp=sharing)

## Acknowledgment
This repo borrows some code from [UP-DETR](https://github.com/dddzg/up-detr) and [point DETR](https://arxiv.org/abs/2104.07434).


