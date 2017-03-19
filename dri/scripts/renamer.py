import sys
from os import listdir
import os

pathIn = sys.argv[1]
name = sys.argv[2]
name2 = sys.argv[3]
pathOut = sys.argv[4]
newName = sys.argv[5]

for fileName in listdir(pathIn):
	if name in fileName and name2 in fileName:
		dest = fileName.replace(name, newName)
		cmd = "cp "+pathIn+"/"+fileName+" "+pathOut+"/"+dest
		print(cmd)
		#os.system(cmd)

#w_no0_LSTM_aug_none_amazon_Music_64500_mNe_10000_mmNe_10_noVl_F_Mem_10_min_5_mEp_200_mNPro_0_trls_10_sFlds_T_PPR_0_onlyPR_F_prT_neutral_bself_none_d_T_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_trial_7_fold_13_BAE_Test.npy