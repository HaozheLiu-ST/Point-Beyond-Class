#!/bin/sh

# ---------------------------------------------------------------------------------------------------------------------------------------
#                    CXR8
# ---------------------------------------------------------------------------------------------------------------------------------------

# exp1 5/10/20/30/40p / 100p  only box (bottom bound) [fcos]
# partial=5
# python3 tools/train.py \
#     configs/fcos/fcos_CXR8.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-CXR8/exp1_fcos512wBB_$[partial]p_onlyBox >  \
#                  /STU_out/logs/mmdet-CXR8/exp1_fcos512wBB_$[partial]p_onlyBox.log 2>&1

# # exp2 5/10/20/30/40/50p_box + pseudo(point DETR)   [fcos]
# partial=5
# python3 tools/train.py \
#     configs/fcos/fcos_CXR8.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-CXR8/exp2_fcos512wBB_$[partial]p__train_LableBox_PseudoBox__exp1_stage1_data$[partial]p_2-3box_baseline >  \
#                  /STU_out/logs/mmdet-CXR8/exp2_fcos512wBB_$[partial]p__train_LableBox_PseudoBox__exp1_stage1_data$[partial]p_2-3box_baseline.log 2>&1
# # exp2 5/10/20/30/40/50p_box + pseudo(our PBC)   [fcos]
# partial=10
# python3 tools/train.py \
#     configs/fcos/fcos_CXR8.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-CXR8/exp2_fcos512wBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth_new >  \
#                  /STU_out/logs/mmdet-CXR8/exp2_fcos512wBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth_new.log 2>&1


# ---------------------------------------------------------------------------------------------------------------------------------
# exp1 5/10/20/30/40p / 100p  only box (bottom bound) [faster]
# partial=5
# python3 tools/train.py \
#     configs/faster/faster_CXR8.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-CXR8/exp1_faster512wBB_$[partial]p_onlyBox >  \
#                  /STU_out/logs/mmdet-CXR8/exp1_faster512wBB_$[partial]p_onlyBox.log 2>&1

# # exp2 5/10/20/30/40/50p_box + pseudo(point DETR)   [faster]
# partial=5
# python3 tools/train.py \
#     configs/faster/faster_CXR8.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-CXR8/exp2_faster512noBB_$[partial]p__train_LableBox_PseudoBox__exp1_wBB_stage1_data$[partial]p_2-3box_baseline >  \
#                  /STU_out/logs/mmdet-CXR8/exp2_faster512noBB_$[partial]p__train_LableBox_PseudoBox__exp1_wBB_stage1_data$[partial]p_2-3box_baseline.log 2>&1

# # exp2 5/10/20/30/40/50p_box + pseudo(our PBC)  [faster]
# partial=5
# python3 tools/train.py \
#     configs/faster/faster_CXR8.py \
#     --seed 42 \
#     --deterministic \
#     --work-dir /STU_out/models/mmdet-CXR8/exp2_faster512wBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth_new >  \
#                  /STU_out/logs/mmdet-CXR8/exp2_faster512wBB_$[partial]p__train_LableBox_PseudoBox__exp4_stage1_data$[partial]p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth_new.log 2>&1
