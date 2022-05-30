import argparse
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve
import numpy as np

def ROC(targets,scores,auc):
    fpr,tpr,_=roc_curve(targets,scores)
    np.savetxt("fpr.txt",fpr)
    np.savetxt("tpr.txt",tpr)
    plt.figure()
    plt.figure(fpr,tpr,color="darkorange",lw=2,label="ROC curve(area=%0.2f)"%(np.mean(auc)))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig("ROC.png")
    plt.show()
    

    
