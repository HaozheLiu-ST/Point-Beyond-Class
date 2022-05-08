#!/bin/sh
## -------------------------------------------------------------------------------------------------------
##                                                    RSNA
## -------------------------------------------------------------------------------------------------------
# | exp 1 | pointDETR baseline | data=5p/10p/20p/30p/40p | train: instances_trainBox.json, test: instances_val.json |
partial=2
OMP_NUM_THREADS=8 python3.6 main.py \
    --epochs 111 \
    --lr_backbone 1e-5 \
    --lr 1e-4 \
    --pre_norm \
    --coco_path /YOURPATH/data/RSNA/cocoAnn/$[partial]p \
    --dataset_file rsna \
    --batch_size 16 \
    --num_workers 16 \
    --data_augment \
    --position_embedding v4 \
    --output_dir ./outfiles/models/RSNA/exp1_stage1_data$[partial]p_baseline >  \
                 ./outfiles/logs/RSNA/exp1_stage1_data$[partial]p_baseline.log 2>&1

# | exp 2 | pointDETR baseline + unlabel_cons | data=5p/10p/20p/30p/40p | point_num=2 consistency loss | train: instances_trainBox.json, test: instances_val.json |
# partial=20
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/RSNA/cocoAnn/$[partial]p \
#     --dataset_file rsna \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --sample_points_num 1  \
#     --train_with_unlabel_imgs \
#     --unlabel_cons_loss_coef 50 \
#     --partial $partial \
#     --output_dir ./outfiles/models/RSNA/exp2_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50 >  \
#                  ./outfiles/logs/RSNA/exp2_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50.log 2>&1

# | exp 3 | pointDETR baseline + 2ptsCons| data=5p/10p/20p/30p/40p | point_num=2 consistency loss | train: instances_trainBox.json, test: instances_val.json |
# partial=50
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/RSNA/cocoAnn/$[partial]p \
#     --dataset_file rsna \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --sample_points_num 2 \
#     --cons_loss \
#     --cons_loss_coef 100 \
#     --output_dir ./outfiles/models/RSNA/exp3_stage1_data$[partial]p_2pts_consLoss100 >  \
#                  ./outfiles/logs/RSNA/exp3_stage1_data$[partial]p_2pts_consLoss100.log 2>&1

# | exp 4 | pointDETR baseline + 2ptsCons(pretrained) + unlabel_cons | data=5p/10p/20p/30p/40p | train: instances_trainBox.json, test: instances_val.json |
# partial=50
# OMP_NUM_THREADS=8 python3.6 main.py \
#     --epochs 111 \
#     --lr_backbone 1e-5 \
#     --lr 1e-4 \
#     --pre_norm \
#     --coco_path /YOURPATH/data/RSNA/cocoAnn/$[partial]p \
#     --dataset_file rsna \
#     --batch_size 16 \
#     --num_workers 16 \
#     --data_augment \
#     --position_embedding v4 \
#     --sample_points_num 1  \
#     --train_with_unlabel_imgs \
#     --unlabel_cons_loss_coef 50 \
#     --partial $partial \
#     --load_from ./outfiles/models/RSNA/exp3_stage1_data$[partial]p_2pts_consLoss100/checkpoint0110.pth \
#     --output_dir ./outfiles/models/RSNA/exp4_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth >  \
#                  ./outfiles/logs/RSNA/exp4_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth.log 2>&1

# gen pseudo labels, note thar coco_path is not important
# OMP_NUM_THREADS=8 python3.6 main.py \
#    --batch_size 1 \
#    --num_workers 8 \
#    --eval \
#    --no_aux_loss \
#    --dataset_file rsna \
#    --pre_norm \
#    --coco_path /YOURPATH/data/RSNA/cocoAnn/40p \
#    --position_embedding v4 \
#    --resume ./outfiles/models/RSNA/exp4_stage1_data50p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth/checkpoint0110.pth \
#    --save_csv /apdcephfs/share_1290796/PointDETR_saveCsv_RSNA/exp4_stage1_data50p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth.csv \
#    --generate_pseudo_bbox \
#    --partial 40
