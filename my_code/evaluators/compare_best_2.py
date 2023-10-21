#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pylab
from mlFunc import mrow
from classifiers import weighted_logistic_reg_score, tied_cov_GC, logistic_reg_calibration
from evaluators.evaluation_GMM import evaluation_GMM_ncomp
from mlFunc import train_SVM_linear, train_SVM_RBF
from validators import *


def bayes_error_plot_2best(D, L, pi, title, ylim):
    
    # #!GMM first
    # bayes_error_min_act_plot(D[0], L, 'GMM', 1.1)
    # #already plots DCF
    
    # #!SVM RBF second
    # bayes_error_min_act_plot(D[1], L, 'SVM_RBF', 1.1)
    
    # ###π = 0.5
    # scores_tot = compute_min_DCF(D[1], L, 0.5, 1, 1)
    
    # t = PrettyTable(["Type", "minDCF"])
    # t.title = "minDCF: π=0.5"
    # t.add_row(['SVM, K=1, C=10, gamma=0.001', round(scores_tot, 3)])
    # print(t)
    
    # #### π = 0.1
    # scores_tot = compute_min_DCF(D[1], L, 0.1, 1, 1)

    # t = PrettyTable(["Type", "minDCF"])
    # t.title = "minDCF: π=0.1"
    # t.add_row(['SVM, K=1, C=10, gamma=0.001', round(scores_tot, 3)])
    # print(t)
    
    # #### π = 0.09
    # scores_tot = compute_min_DCF(D[1], L, 0.9, 1, 1)

    # t = PrettyTable(["Type", "minDCF"])
    # t.title = "minDCF: π=0.9"
    # t.add_row(['SVM, K=1, C=10, gamma=0.001', round(scores_tot, 3)])
    # print(t)
    
    #!Tied MVG
    #bayes_error_min_act_plot(D[2], L, 'MVG_T', 1.1)
    
    # ###π = 0.5
    # llrst_tot = compute_min_DCF(D[2], L, 0.5, 1, 1)  
    # t = PrettyTable(["Type", "minDCF"])
    # t.title = title
    # t.add_row(["MVG tied", round(llrst_tot, 3)])
    # print(t)
    
    # #### π = 0.1
    # llrst_tot = compute_min_DCF(D[2], L, 0.1, 1, 1)  
    # t = PrettyTable(["Type", "minDCF"])
    # t.title = title
    # t.add_row(["MVG tied", round(llrst_tot, 3)])
    # print(t)
    
    # #### π = 0.9
    # llrst_tot = compute_min_DCF(D[2], L, 0.9, 1, 1)  
    # t = PrettyTable(["Type", "minDCF"])
    # t.title = title
    # t.add_row(["MVG tied", round(llrst_tot, 3)])
    # print(t)
 


    #plot min_DCF 
    p = np.linspace(-4, 4, 30)
    pylab.title(title)
    pylab.plot(p, bayes_error_plot(p, D[0], L, minCost=False), color='r', label='GMM_RAW_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[0], L, minCost=True), 'r--', label='GMM_RAW_minDCF')

    pylab.plot(p, bayes_error_plot(p, D[1], L, minCost=False), color='b', label='SVM_RBF_RAW_cal_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[1], L, minCost=True), 'b--', label='SVM_RBF_RAW_cal_minDCF')
    
    pylab.plot(p, bayes_error_plot(p, D[2], L, minCost=False), color='g', label='MVG_T_RAW_actDCF')
    pylab.plot(p, bayes_error_plot(p, D[2], L, minCost=True), 'g--', label='MVG_T_RAW_minDCF')

    pylab.ylim(0, ylim)
    pylab.legend()
    #pylab.savefig('../images/comparison/ROC_2best' + title + '.png')
    pylab.show()

def ROC_2best(D, L, pi, title):
    thresholds = np.array(D[0])
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(D[0] > t)
        conf = confusion_matrix_binary(Pred, L)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR, label="GMM_RAW")

    thresholds = np.array(D[1])
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(D[1] > t)
        conf = confusion_matrix_binary(Pred, L)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR, label="SVM_RBF_RAW_cal")
    
    thresholds = np.array(D[2])
    thresholds.sort()
    thresholds = np.concatenate([np.array([-np.inf]), thresholds, np.array([np.inf])])
    FPR = np.zeros(thresholds.size)
    TPR = np.zeros(thresholds.size)
    for idx, t in enumerate(thresholds):
        Pred = np.int32(D[2] > t)
        conf = confusion_matrix_binary(Pred, L)
        TPR[idx] = conf[1, 1] / (conf[1, 1] + conf[0, 1])
        FPR[idx] = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    pylab.plot(FPR, TPR, label="MVG_T_RAW")

    pylab.title(title)
    pylab.legend()
    #pylab.savefig('../images/comparison/ROC_2best' + title + '.png')
    pylab.show()
    
    
def calibrate_scores(scores, labels):
    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.3):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.3):]

    S, estimated_w, estimated_b = logistic_reg_calibration(np.array([scores_70]), labels_70, np.array([scores_30]), 0, 0.9)

    return np.array(S), labels_30, estimated_w, estimated_b
    
    
    
def compute_2best_plots(DTR, LTR, DTE, LTE):
    
    #!GMM
    GMM_llrst_raw = evaluation_GMM_ncomp('', DTR, LTR, DTE, LTE, 0.5, 4)

    # #!SVM RBF 
    Z = np.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1
    C = 10.0
    K = 1.0
    gamma = 0.001
    aStar, loss = train_SVM_RBF(DTR, LTR, C=C, K=K, gamma=gamma)

    kern = np.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            kern[i, j] = np.exp(-gamma * (np.linalg.norm(DTR[:, i] - DTE[:, j]) ** 2)) + K * K
    SVMRBF_scores = np.sum(np.dot(aStar * mrow(Z), kern), axis=0)

    
    #!MVG Tied
    _, _, MVGT_scores = tied_cov_GC(DTE, DTR, LTR)

    
    #!W LR
    #WLR_scores = weighted_logistic_reg_score(DTR, LTR, DTE, 1e-5)
    #second_scores=[]
    #second_scores.append(WLR_scores)
    
    #!SVM linear
    #C = 10.0
    #K = 1.0
    # wStar, primal, dual, gap = train_SVM_linear(DTR, LTR, C=C, K=K)
    # DTEEXT = np.vstack([DTE, K * np.ones((1, DTE.shape[1]))])
    # second_scores = np.dot(wStar.T, DTEEXT).ravel()

    
    #!calibration
    cal_scores, cal_labels, w, b = calibrate_scores(SVMRBF_scores, LTE)
    SVMRBF_scores = SVMRBF_scores.reshape((1, 6000))
    final_score = np.dot(w.T, SVMRBF_scores) + b
    
    
    # Put here models to be compared
    #bayes_error_plot_2best([D1, D2, D3], sameEval_Labels, 0.5, "", 1.1)
    bayes_error_plot_2best([GMM_llrst_raw, final_score, MVGT_scores], LTE, 0.5, '',1.1)
    ROC_2best([GMM_llrst_raw, final_score, MVGT_scores], LTE, 0.5, '')


    

    


