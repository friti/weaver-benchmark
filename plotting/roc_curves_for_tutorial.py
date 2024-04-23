from sklearn.metrics import roc_curve, auc
import ROOT

fil_mlp = ROOT.TFile.Open("/afs/cern.ch/user/f/friti/work/soft_taus/cmssw/CMSSW_13_0_3/src/weaver-benchmark/condor/output_mlp_20epochs/mlp_predict.root")
tree_mlp = fil_mlp.Get("Events")
fil_ak8 = ROOT.TFile.Open("/afs/cern.ch/user/f/friti/work/soft_taus/cmssw/CMSSW_13_0_3/src/weaver-benchmark/condor/output_deepak8_20epochs/deepak8_predict.root")
tree_ak8 = fil_ak8.Get("Events")
fil_pn = ROOT.TFile.Open("/afs/cern.ch/user/f/friti/work/soft_taus/cmssw/CMSSW_13_0_3/src/weaver-benchmark/condor/output_particlenet_20epochs/particlenet_predict.root")
tree_pn = fil_pn.Get("Events")

score_is_signal = []
is_signal = []
for item in range(tree_mlp.GetEntries()):
    tree_mlp.GetEntry(item)
    score_is_signal.append(tree_mlp.score_is_signal_new)
    is_signal.append(tree_mlp.is_signal_new)

# ROC curves with the sklearn function
fpr_mlp, tpr_mlp, threshold = roc_curve(is_signal, score_is_signal)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

score_is_signal = []
is_signal = []
for item in range(tree_ak8.GetEntries()):
    tree_ak8.GetEntry(item)
    score_is_signal.append(tree_ak8.score_is_signal_new)
    is_signal.append(tree_ak8.is_signal_new)

# ROC curves with the sklearn function
fpr_ak8, tpr_ak8, threshold = roc_curve(is_signal, score_is_signal)
roc_auc_ak8 = auc(fpr_ak8, tpr_ak8)

score_is_signal = []
is_signal = []
for item in range(tree_pn.GetEntries()):
    tree_pn.GetEntry(item)
    score_is_signal.append(tree_pn.score_is_signal_new)
    is_signal.append(tree_pn.is_signal_new)

# ROC curves with the sklearn function
fpr_pn, tpr_pn, threshold = roc_curve(is_signal, score_is_signal)
roc_auc_pn = auc(fpr_pn, tpr_pn)

# Plot with matplotlib
import matplotlib.pyplot as plt
plt.title('')
'''
plt.plot(fpr_mlp, tpr_mlp, 'b', label = 'MLP: AUC = %0.2f' % roc_auc_mlp)
plt.plot(fpr_ak8, tpr_ak8, 'r', label = 'DeepAK8: AUC = %0.2f' % roc_auc_ak8)
plt.plot(fpr_pn, tpr_pn, 'g', label = 'ParticleNet: AUC = %0.2f' % roc_auc_pn)
'''
plt.plot(tpr_mlp, fpr_mlp, 'b', label = 'MLP: AUC = %0.3f' % roc_auc_mlp)
plt.plot(tpr_ak8, fpr_ak8, 'r', label = 'DeepAK8: AUC = %0.3f' % roc_auc_ak8)
plt.plot(tpr_pn, fpr_pn, 'g', label = 'ParticleNet: AUC = %0.3f' % roc_auc_pn)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0.0001, 1])
plt.yscale("log")
plt.xlabel('Signal efficiency')
plt.ylabel('Background efficiency')
plt.savefig("roc_curves_tutorial.png")
plt.savefig("roc_curves_tutorial.pdf")
    










