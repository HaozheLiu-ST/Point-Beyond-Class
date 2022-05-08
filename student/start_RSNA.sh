#!/bin/sh

# ---------------------------------------------------------------------------------------------------------------------------------------
#                    RSNA Pneumonia Detection
# ---------------------------------------------------------------------------------------------------------------------------------------
# exp1 5/10/20/30/40/50p / 100p  only box (bottom bound)  [faster]
# partial=5
# python3 tools/train.py \
#     configs/faster/faster_RSNA.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-RSNA/exp1_faster512_$[partial]p_onlyBox_lr05 >  \
#                  /STU_out/logs/mmdet-RSNA/exp1_faster512_$[partial]p_onlyBox_lr05.log 2>&1

# exp2 10/20/30/40/50p_box + pseudo(point DETR)  [faster]
# partial=30
# python3 tools/train.py \
#     configs/faster/faster_RSNA.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-RSNA/exp2_faster512noBB_$[partial]p__train_LableBox_PseudoBox__exp1_stage1_data$[partial]p_baseline_lr02 >  \
#                  /STU_out/logs/mmdet-RSNA/exp2_faster512noBB_$[partial]p__train_LableBox_PseudoBox__exp1_stage1_data$[partial]p_baseline_lr02.log 2>&1
# exp2 5/10/20/30/40/50p_box + pseudo(our PBC)   [faster]
# partial=50
# python3 tools/train.py \
#     configs/faster/faster_RSNA.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-RSNA/exp2_faster512noBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth200_lr05 >  \
#                  /STU_out/logs/mmdet-RSNA/exp2_faster512noBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth200_lr05.log 2>&1




# ---------------------------------------------------------------------------------------------------------------------------------

# exp1 5/10/20/30/40p / 100p  only box (bottom bound) [fcos]
# partial=100
# python3 tools/train.py \
#     configs/fcos/fcos_RSNA.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-RSNA/exp1_fcos512noBB_$[partial]p_onlyBox_3 >  \
#                  /STU_out/logs/mmdet-RSNA/exp1_fcos512noBB_$[partial]p_onlyBox_3.log 2>&1

# exp2 5/10/20/30/40/50p_box + pseudo(point DETR) [fcos]
# partial=50
# python3 tools/train.py \
#     configs/fcos/fcos_RSNA.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-RSNA/exp2_fcos512noBB_$[partial]p__train_LableBox_PseudoBox__exp1_stage1_data$[partial]p_baseline >  \
#                  /STU_out/logs/mmdet-RSNA/exp2_fcos512noBB_$[partial]p__train_LableBox_PseudoBox__exp1_stage1_data$[partial]p_baseline.log 2>&1
# exp2 5/10/20/30/40/50p_box + pseudo(our PBC)  [fcos]
# partial=50
# python3 tools/train.py \
#     configs/fcos/fcos_RSNA.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-RSNA/exp2_fcos512noBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth200 >  \
#                  /STU_out/logs/mmdet-RSNA/exp2_fcos512noBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth200.log 2>&1

