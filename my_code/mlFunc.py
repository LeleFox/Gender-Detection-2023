import numpy
import matplotlib
import matplotlib.pyplot as plt
import pylab
import scipy.linalg
import sklearn.datasets
import scipy.optimize as opt
from prettytable import PrettyTable

#!tranform in column vector
def mcol(v):
    return v.reshape((v.size, 1))

#!tranform in row vector
def mrow(v):
    return v.reshape((1, v.size))

#!shuffle data 
def randomize(D, L, seed=0):
    nTrain = int(D.shape[1])
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    
    DTR = D[:, idxTrain]
    LTR = L[idxTrain]
    
    return DTR, LTR

#!load data
def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        try:
            for line in f:
                attrs = line.replace(" ", "").split(',')[0:12]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                _label = line.split(',')[-1].strip()
                label = int(_label)
                DList.append(attrs)
                labelsList.append(label)
        except:
            pass
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)



#############################################################
#!Z-normalize data
def znorm(DTR, DTE):
    mu_DTR = mcol(DTR.mean(1))
    std_DTR = mcol(DTR.std(1))

    DTR_z = (DTR - mu_DTR) / std_DTR
    DTE_z = (DTE - mu_DTR) / std_DTR
    return DTR_z, DTE_z

#!Gaussianize data
def gaussianize_features(DTR, DTR_copy):
    P = []
    #for each feature vector
    for dIdx in range(DTR.shape[0]):
        DT = mcol(DTR_copy[dIdx, :]) #transform feature vector into column vector
        X = DTR[dIdx, :] < DT        #compare each sample to each other sample (in the same feature), it returns 1 if it is lower. Notice that it works element by element because you are comparing a column vector to a row vector. 
        R = (X.sum(1) + 1) / (DTR.shape[1] + 2) #compute the RANK (X.sum(1) returns the number of samples LOWER than sample i)
        P.append(scipy.stats.norm.ppf(R))   #the new point in gaussianized space is the value of Probability (on a NORmal gaussian) corresponding to the rank of the point in the original space
    return numpy.vstack(P)
################################################################

#!mean and CovariNCE MATRIX of data (usually of data belonging to 1 class)
def empirical_mean(D):
    return mcol(D.mean(1))

def empirical_covariance(D, mu):
    n = numpy.shape(D)[1]
    DC = D - mcol(mu)
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC))
    return C

################################################################
#!----------------------PCA-----------------
def PCA(D, L, m, filename=None, LDA_flag=False): #D is the dataset, m is the final desired value of dimension (number of final records)
    n = D.shape[1]        #number of samples
    mu = D.mean(axis=1)           #compute the dataset mean over columns (axis=1) of dataset matrix D
    DC = D - mcol(mu)             #center dataset by subtracting to each sample the dataset mean (column vectors)
    
    C = 1 / n * numpy.dot(DC, numpy.transpose(DC)) #compute covariance matrix: C=(Dc*Dc.T)*(1/N)
    
    S, U = numpy.linalg.eigh(C) #compute the eigenvalues and eigenvectors of covariance matrix C
                             #U will contain the eigenvectors  (columns of U) and S the corresponding eigenvalues
                             #already ordered from smallest to greatest
                             #FOR EXAMPLE, THE FIRST EIGENVECTOR IS (0.315487, -0.3197231, -0.479838, 0.753657)
    
    P = U[:, ::-1][:, 0:m] #P will be the direction over which to project our data
                           #it corresponds to the m eigenvectors (associated to the m greatest eigenvalues)
                           #of C matrix.
                           #So, we invert the order of the column vectors of U ([:, ::-1]) 
                           # in order to have them from highest to smallest
                           #and then we take only the first m columns (eigenvectors) corresponding to highest eigenvalues
    

    #Now, we project every data object (column) of dataset D on the direction of P: y=P.T * x
    #DATA PROJECTED over PCA's m dimensions
    
    DPCA = numpy.dot(P.T, D) 
    
    if filename is not None:
        plot_PCA_result(P, DPCA, L, m, filename, LDA_flag)

    #anyway we return the projection matrix, so you need to remember to perform the numpy.dot(P.T,D) in the calling function that calls PCA().
    return P

#!After applying PCA we might want to plot scatter plot (data objects) of each sample, each feature in respect to each other feature.
#!This is because PCA eliminates the directions/features not useful and eliminates correlation on remaining features (so dataset is rotated and looks more sparse (not linearly distributed))
def plot_PCA_result(P, D, L, m, filename, LDA_flag):
    
    plt.figure()
    
    D0 = D[:, L==0] #take every row from D matrix where the corresponding
                     #index element in L array has value 0. 
    D1 = D[:, L==1]
    
    for dIdx1 in range(m-1): 
        for dIdx2 in range(m):  
            if dIdx1 == dIdx2:
                continue
              
            plt.figure()
            plt.xlabel("x") 
            plt.ylabel("y") 
            
            #plot all the attributes of the3 labels in respect to all other attribute
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Male')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Female')
            plt.legend()
            plt.tight_layout()
            
        if LDA_flag is True:
            DTR = numpy.dot(P.T, -D)
            W = LDA(DTR, L, m, k=2) * 100

            plt.quiver(W[0] * -5, W[1] * -5, W[0] * 40, W[1] * 40, units='xy', scale=1, color='g')
            plt.xlim(-65, 65)
            plt.ylim(-25, 25)
            
  
    plt.savefig('./images/' + filename + '.png')
    plt.show()



######################################################################
#!----------------------LDA-----------------


def LDA(D, L, m, k=2):      #D is the dataset, m is the final desired value of dimension (number of final records), k is number of classes
    n = numpy.shape(D)[1]   #compute the dataset mean over columns (axis=1) of dataset matrix D
    mu = D.mean(axis=1)

    #Sb
    Sb=0.0              #Sb= 1/n * Σ n_c*((mu_c-mu)*(mu_c-mu).T)     where n_c is number of samples inside class c, mu_c is mean inner to class C
    for i in range(k):
        class_c= D[:, L==i]
        nc_c= class_c.shape[1] #how many columns (records) I have in class i?  
        mu_c=class_c.mean(axis=1)
        mu_c= mu_c.reshape((mu_c.size, 1))
       
        Sb = Sb + (nc_c * numpy.dot((mu_c-mu), (mu_c-mu).T) )
  
    Sb= Sb / n
    
    #Sw
    Sw = 0
    for i in range(k):      #Sw= 1/n*(n_c * C_c)        where n_c is number of samples inside class c, C_c is covariance matrix inner to class C
        Sw += (L == i).sum() * empirical_covariance(D[:, L == i], empirical_mean(D))

    Sw = Sw / n

    #find W (exatly as P for PCA) direction with eigenvectors
    # associated to maximum eigenvalues of (Sw^-1) * (Sb)
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, 0:m]
    
    
    return W

#########################################################################
#??????????????????????????????????????????????


#generic hystogram plotting of 1 feature, 2 classes used for PCA
def plot_histogram(D, L, labels, title):
    matplotlib.pyplot.figure()
    matplotlib.pyplot.title(title)
    y = D[:, L == 0]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[0])
    y = D[:, L == 1]
    matplotlib.pyplot.hist(y[0], bins=60, density=True, alpha=0.4, label=labels[1])
    matplotlib.pyplot.legend()
    plt.savefig('./images/hist' + title + '.png')
    matplotlib.pyplot.show()

#scatter plot: data points of 1 features for 2 classes
def plot(D, L):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]
    hLabels = {
        0: "male",
        1: "female"
    }
    plt.figure()
    for i in range(2):
        plt.legend(hLabels)
        plt.scatter(D[:, L == i][0], D[:, L == i][1], label=hLabels.get(i))
    plt.show()

######################################################################
#!-----------functions for multivariate Gaussians classifiers-----------------

def ML_GAU(D):
    m = empirical_mean(D)
    C = empirical_covariance(D, m)
    return m, C


def logpdf_GAU_ND(X, mu, C):
    P = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2 * numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]

    Y = []

    for i in range(X.shape[1]):
        x = X[:, i:i + 1]
        res = const + -0.5 * numpy.dot((x - mu).T, numpy.dot(P, (x - mu)))
        Y.append(res)
    return numpy.array(Y).ravel()


def loglikelihood(XND, m_ML, C_ML):
    return logpdf_GAU_ND(XND, m_ML, C_ML).sum()


def likelihood(XND, m_ML, C_ML):
    return numpy.exp(loglikelihood(XND, m_ML, C_ML))



######################################################################
#!-----------functions for Logistic Regression (linear, linear weighted and quadratic)-----------------'''

def logreg_obj_wrap(DTR, LTR, l):
    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, DTR) + b
        cxe = numpy.logaddexp(0, -S * Z)
        return numpy.linalg.norm(w) ** 2 * l / 2.0 + cxe.mean()

    return logreg_obj

def weighted_logreg_obj_wrap(DTR, LTR, l, pi=0.5):
    M = DTR.shape[0]
    Z = LTR * 2 - 1

    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        reg = 0.5 * l * numpy.linalg.norm(w) ** 2
        s = (numpy.dot(w.T, DTR) + b).ravel()
        nt = DTR[:, LTR == 0].shape[1]
        avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
        avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
        return reg + (pi / nt) * avg_risk_1 + (1-pi) / (DTR.shape[1]-nt) * avg_risk_0
    return logreg_obj

def quad_logreg_obj_wrap(DTR, LTR, l, pi=0.5):
    M = DTR.shape[0]
    Z = LTR * 2.0 - 1.0

    #linear not weightet quad log reg
    # def logreg_obj(v):
    #     w = mcol(v[0:M])
    #     b = v[-1]
    #     S = numpy.dot(w.T, DTR) + b
    #     cxe = numpy.logaddexp(0, -S * Z)
    #     return numpy.linalg.norm(w) ** 2 * l / 2.0 + cxe.mean()

    # prior weighted quad log reg
    def logreg_obj(v):
        w = mcol(v[0:M])
        b = v[-1]
        reg = 0.5 * l * numpy.linalg.norm(w) ** 2
        s = (numpy.dot(w.T, DTR) + b).ravel()
        nt = DTR[:, LTR == 0].shape[1]
        avg_risk_0 = (numpy.logaddexp(0, -s[LTR == 0] * Z[LTR == 0])).sum()
        avg_risk_1 = (numpy.logaddexp(0, -s[LTR == 1] * Z[LTR == 1])).sum()
        return reg + (pi / nt) * avg_risk_1 + (1-pi) / (DTR.shape[1]-nt) * avg_risk_0

    return logreg_obj


######################################################################
#!-----------functions for Support Vector Machines-----------------'''

def calculate_lbgf(H, DTR, C):
    def JDual(alpha):
        Ha = numpy.dot(H, mcol(alpha))
        aHa = numpy.dot(mrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)

    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds=[(0, C)] * DTR.shape[1],
        factr=1.0,
        maxiter=10000,
        maxfun=100000,
    )

    return alphaStar, JDual, LDual


def train_SVM_linear(DTR, LTR, C, K=1):
    DTREXT = numpy.vstack([DTR, K * numpy.ones((1, DTR.shape[1]))])
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.dot(DTREXT.T, DTREXT)
    H = mcol(Z) * mrow(Z) * H

    def JPrimal(w):
        S = numpy.dot(mrow(w), DTREXT)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * numpy.linalg.norm(w) ** 2 + C * loss

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)
    wStar = numpy.dot(DTREXT, mcol(alphaStar) * mcol(Z))
    return wStar, JPrimal(wStar)


def train_SVM_polynomial(DTR, LTR, C, K=1, constant=0, degree=2):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = (numpy.dot(DTR.T, DTR) + constant) ** degree + K ** 2
    # Dist = mcol((DTR**2).sum(0)) + mrow((DTR**2).sum(0)) - 2*numpy.dot(DTR.T, DTR)
    # H = numpy.exp(-Dist)
    H = mcol(Z) * mrow(Z) * H

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]


def train_SVM_RBF(DTR, LTR, C, K=1, gamma=1.):
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    # kernel function
    kernel = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            kernel[i, j] = numpy.exp(-gamma * (numpy.linalg.norm(DTR[:, i] - DTR[:, j]) ** 2)) + K * K
    H = mcol(Z) * mrow(Z) * kernel

    alphaStar, JDual, LDual = calculate_lbgf(H, DTR, C)

    return alphaStar, JDual(alphaStar)[0]

def get_DTRs(DTR, LTR, number_of_classes):

    DTRs = []
    for i in range(number_of_classes):
        DTRs.append(DTR[:, LTR == i])
    return DTRs

################################################################
#! compute accuracy
def test(LTE, LPred):
    accuracy = (LTE == LPred).sum() / LTE.size
    error = 1 - accuracy
    return accuracy, error