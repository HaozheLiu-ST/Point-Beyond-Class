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
'./outfiles/logs/RSNA/exp1_stage1_data5p_baseline.log|r',  # 2.1
'./outfiles/logs/RSNA/exp1_stage1_data10p_baseline.log|g', # 2.8
'./outfiles/logs/RSNA/exp1_stage1_data20p_baseline.log|b', # 9.0
'./outfiles/logs/RSNA/exp1_stage1_data30p_baseline.log|c', # 18.8
'./outfiles/logs/RSNA/exp1_stage1_data40p_baseline.log|k', # 23.7
'./outfiles/logs/RSNA/exp1_stage1_data50p_baseline.log|y', # 25.1
'./outfiles/logs/RSNA/exp3_stage1_data5p_2pts_consLoss100.log|r^-', # 3.4
'./outfiles/logs/RSNA/exp3_stage1_data10p_2pts_consLoss100.log|g^-', # 9.5
'./outfiles/logs/RSNA/exp3_stage1_data20p_2pts_consLoss100.log|b^-', # 18.3
'./outfiles/logs/RSNA/exp3_stage1_data30p_2pts_consLoss200.log|c^-', # 23.8
'./outfiles/logs/RSNA/exp3_stage1_data40p_2pts_consLoss100.log|k^-', # 25.6
'./outfiles/logs/RSNA/exp3_stage1_data50p_2pts_consLoss100.log|y^-', # 29.1
'./outfiles/logs/RSNA/exp2_stage1_data5p_1pts_Erase20_jit005_unlabelLossL2Loss50|r-.', # 4.1
'./outfiles/logs/RSNA/exp2_stage1_data10p_1pts_Erase20_jit005_unlabelLossL2Loss50|g-.', # 8.4
'./outfiles/logs/RSNA/exp2_stage1_data20p_1pts_Erase20_jit005_unlabelLossL2Loss50|b-.', # 14.8
'./outfiles/logs/RSNA/exp2_stage1_data30p_1pts_Erase20_jit005_unlabelLossL2Loss50|c-.', # 24.6
'./outfiles/logs/RSNA/exp2_stage1_data40p_1pts_Erase20_jit005_unlabelLossL2Loss50|k-.', # 30.6
'./outfiles/logs/RSNA/exp2_stage1_data50p_1pts_Erase20_jit005_unlabelLossL2Loss50.log|y-.', # 32.4
'./outfiles/logs/RSNA/exp4_stage1_data5p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth.log|r*-', # 6.8
'./outfiles/logs/RSNA/exp4_stage1_data10p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth.log|g*-', # 17.8
'./outfiles/logs/RSNA/exp4_stage1_data20p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth.log|b*-', # 28.1
'./outfiles/logs/RSNA/exp4_stage1_data30p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth200.log|c*-', # 34.4
'./outfiles/logs/RSNA/exp4_stage1_data40p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth500.log|k*-', # 36.9
'./outfiles/logs/RSNA/exp4_stage1_data50p_1pts_Erase20_jit005_unlabelLossL2Loss50_Load2ptsconsPth.log|y*-', # 39.2
]




plt.xlabel('epochs')
plt.ylabel('AP_50')
plt.title('epoch-AP50')
for i, logFile in enumerate(logFiles):
    epochs, APs, color = parseLog(logFile)
    plt.plot(epochs, APs, color)
plt.savefig('visRSNA.jpg')
