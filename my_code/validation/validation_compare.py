# -*- coding: utf-8 -*-
import sys
import numpy as np

from evaluators.compare_best_2 import bayes_error_plot_2best, ROC_2best
from evaluators.evaluation_GMM import evaluation_GMM_ncomp, evaluation_scores_GMM_ncomp

sys.path.append('../')
from validators import *

def calibrate_scores(scores, labels):
    scores_70 = scores[:int(len(scores) * 0.7)]
    scores_30 = scores[int(len(scores) * 0.7):]
    labels_70 = labels[:int(len(labels) * 0.7)]
    labels_30 = labels[int(len(labels) * 0.7):]

    S, estimated_w, estimated_b = logistic_reg_calibration(numpy.array([scores_70]), labels_70, numpy.array([scores_30]), 0)

    return numpy.array(S), labels_30, estimated_w, estimated_b


def compare(scores, scores2, LTE):
    scores = np.hstack(scores)
    scores2 = np.hstack(scores2)

    #first
    bayes_error_min_act_plot(scores, LTE, 'Calibrated SVM_RBF K=1 C=10 gamma=0.001', 1.1)
    
    #second
    bayes_error_min_act_plot(scores2, LTE, 'Calibrated MVG_Tied', 1.1)

    #compare
    #bayes_error_plot_2best([scores, scores2], LTE, 0.5, '', 1.1)
    #ROC_2best([scores, scores2], LTE, 0.5, '')
    
    
    #At the end, just print the min_DCFs
    # t = PrettyTable(["Type", "π=0.5", "π=0.1", "π=0.9"])
    # t.title = appendToTitle
    # t.add_row(['GMM 4comp' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    # print(t)
    # t.add_row(['WEIGHTED_LR, lambda=' + str(l) + " π_t=" + str(pi), round(scores_tot_05, 3), round(scores_tot_01, 3), round(scores_tot_09, 3)])
    # print(t)




def kfold_validation_compare(DTR, LTR, l=None):
    k = 5
    Dtr = numpy.split(DTR, k, axis=1)
    Ltr = numpy.split(LTR, k)

    scores_append = []
    scores2_append = []
    LR_labels = []

    for i in range(k):
        D = []
        L = []
        if i == 0:
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == k - 1:
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]
        Lte = Ltr[i]
        
        #!SVM RBF 
        Z = L * 2 - 1
        C = 10.0
        K = 1.0
        gamma=0.001
        aStar, loss = train_SVM_RBF(D, L, C=C, K=K, gamma=0.001)
        kern = numpy.zeros((D.shape[1], Dte.shape[1]))
        for i in range(D.shape[1]):
            for j in range(Dte.shape[1]):
                kern[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(D[:, i] - Dte[:, j]) ** 2)) + K * K
        scores = numpy.sum(numpy.dot(aStar * mrow(Z), kern), axis=0)
        scores_append.append(scores)

        #!GMM
        GMM_llrst_raw = evaluation_scores_GMM_ncomp('', D, L, Dte, Lte, 0.5, 4)
        #scores_append.append(GMM_llrst_raw)
        
        
        #!mvg TIED COV
        _, _, llrs_tied = tied_cov_GC(Dte, D, L)
        scores2_append.append(llrs_tied)
        
        #!Weighted logistic regression
        #WLR_scores = weighted_logistic_reg_score(D, L, Dte, 1e-4, pi=0.5)
        #scores2_append.append(WLR_scores)
        
        
        LR_labels = np.append(LR_labels, Lte, axis=0)
        LR_labels = np.hstack(LR_labels)
        
        
        ## CALIBRATION INSIDE
        # WLR_scores = WLR_scores.reshape((1, 800))
        # L = L.reshape((1, 1600))
        # S, estimated_w, estimated_b = logistic_reg_calibration(WLR_scores, Lte, L, 0) 
        # final_score = numpy.dot(estimated_w.T, WLR_scores) + estimated_b
        # scores2_append.append(final_score)
        

        

    ## CALIBRATION outside
    
    scores_append = np.hstack(scores_append)
    cal_scores, cal_labels, w1, b1 = calibrate_scores(scores_append, LR_labels)
    scores_append = scores_append.reshape((1, 2400))
    final_score1 = numpy.dot(w1.T, scores_append) + b1
    
    scores2_append = np.hstack(scores2_append)
    cal_scores, cal_labels, w2, b2 = calibrate_scores(scores2_append, LR_labels)
    scores2_append = scores2_append.reshape((1, 2400))
    final_score2 = numpy.dot(w2.T, scores2_append) + b2
    
  
    compare(final_score1, final_score2, LR_labels)


def compare_2_validation(DTR, LTR, L):
    for l in L:
        kfold_validation_compare(DTR, LTR, l)
