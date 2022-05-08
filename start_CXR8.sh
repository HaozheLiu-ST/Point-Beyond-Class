#!/bin/sh
## -------------------------------------------------------------------------------------------------------
##                                                    CXR 8(后7类合并为一类)
## -------------------------------------------------------------------------------------------------------
# | exp 1 | pointDETR baseline | data=5p/10p/20p/30p/40p | train: instances_trainBox.json, test: instances_val.json |
# partial=10
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/$[partial]p \
#     --dataset_file cxr8 \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --output_dir ./outfiles/models/CXR8/exp1_stage1_data$[partial]p_2-3box_baseline >  \
#                  ./outfiles/logs/CXR8/exp1_stage1_data$[partial]p_2-3box_baseline.log 2>&1

# | exp 2 | pointDETR baseline | data=5p/10p/20p/30p/40p | point_num=2 consistency loss | train: instances_trainBox.json, test: instances_val.json |
# partial=20
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/$[partial]p \
#     --dataset_file cxr8 \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --sample_points_num 2 \
#     --cons_loss \
#     --cons_loss_coef 100 \
#     --output_dir ./outfiles/models/CXR8/exp2_stage1_data$[partial]p_2-3box_2pts_consLoss100 >  \
#                  ./outfiles/logs/CXR8/exp2_stage1_data$[partial]p_2-3box_2pts_consLoss100.log 2>&1

# exp 3  data=5p/10p/20p/30p/40p unlabel_cons
# partial=20
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/$[partial]p \
#     --dataset_file cxr8 \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --sample_points_num 1  \
#     --train_with_unlabel_imgs \
#     --unlabel_cons_loss_coef 50 \
#     --partial $partial \
#     --output_dir ./outfiles/models/CXR8/exp3_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box >  \
#                  ./outfiles/logs/CXR8/exp3_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log 2>&1

# # exp 4  data=5p/10p/20p/30p/40p unlabel_cons   有/无（跟exp9_一样） 加载box两个点+cons loss训练的权重
# partial=10
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/$[partial]p \
#     --dataset_file cxr8 \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --sample_points_num 1  \
#     --train_with_unlabel_imgs \
#     --unlabel_cons_loss_coef 50 \
#     --partial $partial \
#     --load_from ./outfiles/models/CXR8/exp2_stage1_data$[partial]p_2-3box_2pts_consLoss100/checkpoint0110.pth \
#     --output_dir ./outfiles/models/CXR8/exp4_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth >  \
#                  ./outfiles/logs/CXR8/exp4_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log 2>&1

# gen pseudo labels, note thar coco_path is not important
OMP_NUM_THREADS=8 python3.6 main.py \
   --batch_size 1 \
   --num_workers 8 \
   --eval \
   --no_aux_loss \
   --dataset_file cxr8 \
   --pre_norm \
   --coco_path /YOURPATH/data/CXR/ClsAll8_cocoAnnWBF/10p \
   --position_embedding v4 \
   --resume ./outfiles/models/CXR8/exp4_stage1_data5p_1pts_Erase20_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth/checkpoint0110.pth \
   --save_csv /apdcephfs/share_1290796/PointDETR_saveCsv_CXR8/exp4_stage1_data5p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth_new.csv \
   --generate_pseudo_bbox \
   --partial 20 > /YOURPATH/logs/tmp1.log
