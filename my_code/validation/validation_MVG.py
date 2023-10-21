# -*- coding: utf-8 -*-
import sys
import numpy as np

sys.path.append('../')
from mlFunc import *
from validators import *
from classifiers import *
from classifiers import *
from validators import *
from prettytable import PrettyTable
from plot_features import plot_features


def compute_MVG_score(Dte, D, L, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels):
    _, _, llrs = MVG(Dte, D, L)             #log-likelihood ratio multivariate gausian for each sample
    _, _, llrsn = naive_MVG(Dte, D, L)      #log-likelihood ratio- naive for each sample
    _, _, llrst = tied_cov_GC(Dte, D, L)    #log-likelihood ratio- tied cov. for each sample
    _, _, llrsnt = tied_cov_naive_GC(Dte, D, L)   #log-likelihood ratio- naive+tied cov. for each sample

    MVG_res.append(llrs)
    MVG_naive.append(llrsn)
    MVG_t.append(llrst)
    MVG_nt.append(llrsnt)
    
    return MVG_res, MVG_naive, MVG_t, MVG_nt


def evaluation(title, pi, MVG_res, MVG_naive, MVG_t, MVG_nt, MVG_labels, appendToTitle):
    MVG_res = np.hstack(MVG_res)
    MVG_naive = np.hstack(MVG_naive)
    MVG_t = np.hstack(MVG_t)
    MVG_nt = np.hstack(MVG_nt)
    
    #!compute_min_DCF receives all scores for that model, and will compute a minDCF by searching for optimal threshold
    llrs_tot = compute_min_DCF(MVG_res, MVG_labels, pi, 1, 1)
    llrsn_tot = compute_min_DCF(MVG_naive, MVG_labels, pi, 1, 1)
    llrst_tot = compute_min_DCF(MVG_t, MVG_labels, pi, 1, 1)
    llrsnt_tot = compute_min_DCF(MVG_nt, MVG_labels, pi, 1, 1)

    # plot_ROC(MVG_res, MVG_labels, appendToTitle + 'MVG')
    # plot_ROC(MVG_naive, MVG_labels, appendToTitle + 'MVG + Naive')
    # plot_ROC(MVG_t, MVG_labels, appendToTitle + 'MVG + Tied')
    # plot_ROC(MVG_nt, MVG_labels, appendToTitle + 'MVG + Naive + Tied')

    # # Cfn and Ctp are set to 1
    # bayes_error_min_act_plot(MVG_res, MVG_labels, appendToTitle + 'MVG', 0.4)
    # bayes_error_min_act_plot(MVG_naive, MVG_labels, appendToTitle + 'MVG + Naive', 1)
    bayes_error_min_act_plot(MVG_t, MVG_labels, appendToTitle + 'MVG + Tied', 1.1)
    # bayes_error_min_act_plot(MVG_nt, MVG_labels, appendToTitle + 'MVG + Naive + Tied', 1)

    #At the end, just print the minDCF
    t = PrettyTable(["Type", "minDCF"])
    t.title = title
    t.add_row(["MVG", round(llrs_tot, 3)])
    t.add_row(["MVG naive", round(llrsn_tot, 3)])
    t.add_row(["MVG tied", round(llrst_tot, 3)])
    t.add_row(["MVG naive + tied", round(llrsnt_tot, 3)])
    print(t)






def validation_MVG(DTR, LTR, appendToTitle, PCA_Flag=True, Gauss_flag = False, zscore=False):
    
    MVG_labels = []
    
    #no PCA
    MVG_res = []        #full covariance
    MVG_naive = []      #naive 
    MVG_t = []          #tied covarinace
    MVG_nt = []         #naive + tied
    
    #first PCA (m=11)
    PCA_mvg = []         
    PCA_mvg_naive = []
    PCA_mvg_t = []
    PCA_mvg_nt = []

    #second PCA (m=10)
    PCA2_mvg = []
    PCA2_mvg_naive = []
    PCA2_mvg_t = []
    PCA2_mvg_nt = []
    
    #! K-fold cross validation: given the limited number of evaluation samples, we turn our attention to cross-validation approaches. 
    #! The K-fold cross validation method can be employed to split the dataset in K, non overlapping, subsets. We then iteratively consider one subset as evaluation, 
    #! and the remaining K− 1 as training set. We can then accumulate the number of errors and compute a global accuracy.
    K = 5
    Dtr = numpy.split(DTR, K, axis=1)   #split training set into k=5 parts (axis=1-->columns/samples)
    Ltr = numpy.split(LTR, K)
    
    for i in range(K):                  #for each fold, select 1 fold for evaluation (as it was a test set) and remaining k-1 folds for validation (modeling)
        
        D = []                          #D[] and L[] will contain, at each iterations, the K-1 folds to be used for validation. 
        L = []
        if i == 0:                              #at first iteration you leave fold 0 and insert into D[] and L[] from fold 1 on...
            D.append(np.hstack(Dtr[i + 1:])) 
            L.append(np.hstack(Ltr[i + 1:]))
        elif i == K - 1:                        #at last iteration you leave the last fold i 
            D.append(np.hstack(Dtr[:i]))
            L.append(np.hstack(Ltr[:i]))
        else:                                   #in all other iterations you put in current D[]: from 0 to i, from i+1 to last. So i_th fold will be used for evaluation
            D.append(np.hstack(Dtr[:i]))
            D.append(np.hstack(Dtr[i + 1:]))
            L.append(np.hstack(Ltr[:i]))
            L.append(np.hstack(Ltr[i + 1:]))

        D = np.hstack(D)
        L = np.hstack(L)

        Dte = Dtr[i]                #i_th fold will be used for evaluation
        Lte = Ltr[i]

        if (zscore):
            D, Dte = znorm(D, Dte)

        if (Gauss_flag):
            D_training = D
            D = gaussianize_features(D, D)
            Dte = gaussianize_features(D_training, Dte)


        #at each iteration we update MVG_labels with the labels of the left out fold.
        #these labels will be compared with the predicted ones in evaluation step at th eend of the for()
        MVG_labels = np.append(MVG_labels, Lte, axis=0)
        MVG_labels = np.hstack(MVG_labels)
        
        
        # Once we have computed our folds, we train the model 
        # (train the model using D and L of this iteration and compute scores for each sample in evaluation FOLD of this iteration)

        #each of these functions will update the respective score matrix by appending the scores for the left out
        #fold of the i_th iteration!
        MVG_res, MVG_naive, MVG_t, MVG_nt = compute_MVG_score(
            Dte,
            D,
            L,
            MVG_res,
            MVG_naive,
            MVG_t,
            MVG_nt,
            MVG_labels)

        if PCA_Flag is True:
            
            # PCA m=11
            P = PCA(D, L, m=11)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA_mvg, PCA_mvg_naive, PCA_mvg_t, PCA_mvg_nt = compute_MVG_score(
                DTE_PCA,
                DTR_PCA,
                L,
                PCA_mvg,
                PCA_mvg_naive,
                PCA_mvg_t,
                PCA_mvg_nt,
                MVG_labels)

            # PCA m=10
            P = PCA(D, L, m=10)
            DTR_PCA = numpy.dot(P.T, D)
            DTE_PCA = numpy.dot(P.T, Dte)

            PCA2_mvg, PCA2_mvg_naive, PCA_2mvg_t, PCA2_mvg_nt = compute_MVG_score(
                DTE_PCA,
                DTR_PCA,
                L,
                PCA2_mvg,
                PCA2_mvg_naive,
                PCA2_mvg_t,
                PCA2_mvg_nt,
                MVG_labels)


    #! At this point of the for, we trained the model for k-1 folds and we want to use the left out 1 fold to evaluate the model over different applications
    #! with different priors. As model performance parameter we use the minDCF which is the cost that we would pay if we knew in advance the optimal threshold 
    #! (so, if we knew in advance the labels in evaluation set). In other words, this is the BEST performance we would achieve, which is better than the real Actual_DCF.

    # π = 0.5 (our application prior)
    evaluation("minDCF: π=0.5", 0.5, 
               MVG_res, 
               MVG_naive, 
               MVG_t, 
               MVG_nt, 
               MVG_labels, 
               appendToTitle + 'minDCF_π=0.5__')

    ###############################

    # π = 0.1
    evaluation("minDCF: π=0.1", 0.1, 
               MVG_res, 
               MVG_naive, 
               MVG_t, 
               MVG_nt, 
               MVG_labels, 
               appendToTitle + 'minDCF_π=0.1__')

    ###############################

    # π = 0.9
    evaluation("minDCF: π=0.9", 0.9, 
               MVG_res, 
               MVG_naive, 
               MVG_t, 
               MVG_nt, 
               MVG_labels, 
               appendToTitle + "minDCF_π=0.9__")

    if PCA_Flag is True:
        
        #! PCA m=11
        # π = 0.5 
        evaluation("minDCF: π=0.5 | PCA m=11", 0.5, 
                   PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.5_PCA m=11__")

        # π = 0.1
        evaluation("minDCF: π=0.1 | PCA m=11", 0.1, 
                   PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.1_PCA m=11__")

        
        # π = 0.9
        evaluation("minDCF: π=0.9 | PCA m=11", 0.9, 
                   PCA_mvg,
                   PCA_mvg_naive,
                   PCA_mvg_t,
                   PCA_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.9_PCA m=11__")
        
        #! PCA m=10
        # π = 0.5 
        evaluation("minDCF: π=0.5 | PCA m=10", 0.5, 
                   PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.5_PCA m=10__")


        # π = 0.1
        evaluation("minDCF: π=0.1 | PCA m=10", 0.1, 
                   PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.1_PCA m=10__")


        # π = 0.9
        evaluation("minDCF: π=0.9 | PCA m=10", 0.9, 
                   PCA2_mvg,
                   PCA2_mvg_naive,
                   PCA2_mvg_t,
                   PCA2_mvg_nt,
                   MVG_labels, appendToTitle + "minDCF_π=0.9_PCA m=10__")

