import numpy as np

name = "../experiments/models/w_no0_LSTMwMini_aug_none_preprocessed_facebook_mNe_10000_mmNe_10_noVl_F_Mem_5_min_20_mEp_200_mNPro_100_trls_10_sFlds_T_PPR_True_onlyPR_F_prT_neg_bself__d_F_lim_F_rinit_F_p_F_pro_F_lH_F_lR_F_trial_0_fold_13_ACC_Test.npy"
TestSet = ["Test", "TestC"]
trials = range(0, 10)
for tSet in TestSet:
    xList = []
    for i in trials:
        newName = name.replace("trial_0", "trial_"+str(i)).replace("_Test.npy", "_"+tSet+".npy")
        x = np.load(newName)
        xList.append(x[0])

    xArr = np.array(xList)
    print(tSet+"-- AVG: "+str(np.mean(xArr))+"STD: "+str(np.std(xArr)))
