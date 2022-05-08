from math import e
import matplotlib.pyplot as plt
import os
import numpy as np

def parseLog(logFile=None):
    logFile, color = logFile.split('|')
    with open(logFile) as f:
        lines = f.readlines()

    APs = []
    idx = 0

    epochs = []
    cur_epoch = 0
    for idx in range(len(lines)):
        if 'Average Precision  (AP) @[ IoU=0.50   ' in lines[idx]:
            cur_ap = 100*float(lines[idx].strip().split()[-1])
            APs.append(cur_ap)
            epochs.append(cur_epoch)
        elif 'Epoch: [' in lines[idx]:
            cur_epoch = int(lines[idx].strip().split('Epoch: [')[-1].split(']')[0])

    if len(epochs) != len(APs):
        return epochs + [epochs[-1]+1], APs, color
    else:
        return epochs, APs, color

logFiles = [
'./outfiles/logs/CXR8/exp1_stage1_data5p_2-3box_baseline.log|r', # 8.9
'./outfiles/logs/CXR8/exp1_stage1_data10p_2-3box_baseline.log|g', # 9.5
'./outfiles/logs/CXR8/exp1_stage1_data20p_2-3box_baseline.log|b', # 10.1
'./outfiles/logs/CXR8/exp1_stage1_data30p_2-3box_baseline.log|c', # 10.6
'./outfiles/logs/CXR8/exp1_stage1_data40p_2-3box_baseline.log|k', # 10.9
'./outfiles/logs/CXR8/exp1_stage1_data50p_2-3box_baseline.log|y', # 12.3
'./outfiles/logs/CXR8/exp2_stage1_data5p_2-3box_2pts_consLoss100.log|r^-', # 9.4
'./outfiles/logs/CXR8/exp2_stage1_data10p_2-3box_2pts_consLoss100.log|g^-', # 9.3
'./outfiles/logs/CXR8/exp2_stage1_data20p_2-3box_2pts_consLoss100.log|b^-', # 13.4
'./outfiles/logs/CXR8/exp2_stage1_data30p_2-3box_2pts_consLoss100.log|c^-', # 15.8
'./outfiles/logs/CXR8/exp2_stage1_data40p_2-3box_2pts_consLoss100.log|k^-', # 20.1
'./outfiles/logs/CXR8/exp2_stage1_data50p_2-3box_2pts_consLoss100.log|y^-', # 21.9
'./outfiles/logs/CXR8/exp3_stage1_data5p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log|r-.', # 8.0
'./outfiles/logs/CXR8/exp3_stage1_data10p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log|g-.', # 9.6
'./outfiles/logs/CXR8/exp3_stage1_data20p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log|b-.', # 9.9
'./outfiles/logs/CXR8/exp3_stage1_data30p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log|c-.', # 10.5
'./outfiles/logs/CXR8/exp3_stage1_data40p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log|k-.', # 12.4
'./outfiles/logs/CXR8/exp3_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box.log|y-.', # 11.8
'./outfiles/logs/CXR8/exp4_stage1_data5p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log|r*-', # 9.5
'./outfiles/logs/CXR8/exp4_stage1_data10p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log|g*-', # 13.0
'./outfiles/logs/CXR8/exp4_stage1_data20p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log|b*-', # 15.4
'./outfiles/logs/CXR8/exp4_stage1_data30p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log|c*-', # 21.9
'./outfiles/logs/CXR8/exp4_stage1_data40p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log|k*-', # 27.5
'./outfiles/logs/CXR8/exp4_stage1_data50p_1pts_Erase0_jit005_unlabelLossL2Loss50_2-3box_Load2ptsconsPth.log|y*-', # 28.5
]




plt.xlabel('epochs')
plt.ylabel('AP_50')
plt.title('epoch-AP50')
for i, logFile in enumerate(logFiles):
    epochs, APs, color = parseLog(logFile)
    plt.plot(epochs, APs, color)
plt.savefig('visCXR8.jpg')
