# -*- coding: utf-8 -*-
'''
Created on Tue Mar 15 09:17:02 2022
@author: delicado
Simple example of Relevance by Ghost Variables.
Code arranged from 'example.py' in 'pyhrt'
'''

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from scipy.stats import norm
#from sklearn.linear_model import LinearRegression
#from collections import defaultdict
#from pyhrt.utils import p_value_2sided
#from sklearn import linear_model
from tools_from_pyhrt.continuous import fit_mdn#, GaussianMixtureModel, MixtureDensityNetwork

def generating_model(N, P=[50,50,50,50]):
# Simulated example 3
# For j=1,...,4{
#   (xj.1,...,xj.pj) normals with sd sigma.j and correlations rho.j
#   zj = xj.1+...+xj.pj
# }
# where rho.1 = rho.3 = 0, so variables in blocks x1 and x3 are independent.
# Moreover blocks x1, x2, x3 and x4 are independent. 
# So z1, z3, z4, z2 are independent.
#
# Response variable:
# y = beta1*z1 + beta2*z2 + beta3*z1 + beta4*z2 + epsilon2
# with epsilon ~ N(0,sigma.eps^2)
# and beta3=beta4=0, so variables in blocks x3 and x4 are irrelevant for y.
#
# linear model to be fitted:
# y ~ x1.1+...+x1.p1 + x2.1+...+x2.p2 + x3.1+...+x3.p3 + x4.1+...+x4.p4
#
    p1 = P[0] # number of uncorrelated variables relevant for y
    p2 = P[1] # number of correlated variables relevant for y
    p3 = P[2] # number of uncorrelated variables irrelevant for y
    p4 = P[3] # number of correlated variables irrelevant for y
    
    sigma_1 = 1 # sd for the p1 variables x1_1,...,x1.p1
    sigma_2 = 1 # sd for the p2 variables x2.1,...,x2.p2
    sigma_3 = 1 # sd for the p3 variables x3.1,...,x3.p3
    sigma_4 = 1 # sd for the p4 variables x4.1,...,x4.p4
    
    sigma_eps = 1 # residual sd for defining y
    
    # rho.1 = rho.3 = 0
    rho_2 = .95 # correlation between p2 variables
    rho_4 = .95 # correlation between p4 variables
    
    beta1 = .5 # coef. of z1=x1.1+...+x1.p1
    beta2 = 1  # coef. of z2=x2.1+...+x2.p2
    beta3 = 0  # coef. of variables in X3
    beta4 = 0  # coef. of variables in X4
    
    # Generating the p1 variables
    X1 = np.random.normal(0,sigma_1,size=(N,p1))
    z1 = X1.dot(np.ones(p1))
    
    # Generating the p2 variables
    Sigma_2 = rho_2 * np.ones((p2,p2)) + (1-rho_2)*np.identity(p2)
    eig_Sigma_2 = np.linalg.eigh(Sigma_2)
    eig_val = eig_Sigma_2[0]
    eig_vec = eig_Sigma_2[1]
    sqrt_Sigma_2 = eig_vec.dot( np.diagflat(np.sqrt(eig_val))).dot(eig_vec.T)

    X2 = np.random.normal(0,sigma_2,size=(N,p2)).dot(sqrt_Sigma_2)
    z2 = X2.dot(np.ones(p2))
    
    # Generating the p3 variables
    X3 = np.random.normal(0,sigma_3,size=(N,p3))
    z3 = X3.dot(np.ones(p3))
    
    # Generating the p4 variables
    Sigma_4 = rho_4 * np.ones((p4,p4)) + (1-rho_4)*np.identity(p4)
    eig_Sigma_4 = np.linalg.eigh(Sigma_4)
    eig_val = eig_Sigma_4[0]
    eig_vec = eig_Sigma_4[1]
    sqrt_Sigma_4 = eig_vec.dot( np.diagflat(np.sqrt(eig_val))).dot(eig_vec.T)

    X4 = np.random.normal(0,sigma_4,size=(N,p4)).dot(sqrt_Sigma_4)
    z4 = X4.dot(np.ones(p4))
    
    X = np.concatenate((X1,X2,X3,X4),axis=1)
    
    # defining the response variable
    y = beta1*z1 + beta2*z2 + beta3*z3 + beta4*z4 + np.random.normal(0,sigma_eps,size=(N))

    return X, y


def generating_model_Hooker(N, r=0.9):
    #### from Hooker et al :   
    #n1 <- 2000 # size of the training sample 
    #n2 <- 1000 # size of the test sample
    #nsim = 50
    nfeat  = 10
    # generating the explanatory variables
    X = np.random.uniform(size=(N,nfeat))
    R = np.array([[1, r], [0, np.sqrt(1-r**2)]])
    X[:,:2] = norm.cdf(norm.ppf(X[:,:2]).dot(R))
    ytrue = 1*X[:,0] + 1*X[:,1] + X[:,2:5].sum(axis=1) + \
         0.5*X[:,6] + 0.8*X[:,7] + 1.2*X[:,8] + 1.5*X[:,9] 
    y = ytrue + 0.1*np.random.normal(size=(N,))
    return X, y

def generating_model_2022(N, P=[5,45,50]):
# Simulated example 3
# For j=1,2,3{
#   (xj.1,...,xj.pj) normals with sd sigma.j and correlations rho.j
#   zj = xj.1+...+xj.pj
# }
# where rho.1 = 0, so variables in block x1  are uncorrelated.
# Moreover blocks x1, x2 and x3 are independent. 
# So z1, z2, z3 are independent.
#
# Response variable:
# y = beta1*z1 + beta2*z2 + beta3*z1 + epsilon2
# with epsilon ~ N(0,sigma.eps^2)
# and beta3=0, so variables in blocks x3 are irrelevant for y.
#
# linear model to be fitted:
# y ~ x1.1+...+x1.p1 + x2.1+...+x2.p2 + x3.1+...+x3.p3
#
    p1 = P[0] # number of uncorrelated variables relevant for y
    p2 = P[1] # number of correlated variables relevant for y
    p3 = P[2] # number of correlated variables irrelevant for y
    
    sigma_1 = 1 # sd for the p1 variables x1_1,...,x1.p1
    sigma_2 = 1 # sd for the p2 variables x2.1,...,x2.p2
    sigma_3 = 2 # sd for the p3 variables x3.1,...,x3.p3
    
    sigma_eps = 1 # residual sd for defining y
    
    # rho.1 = 0
    rho_2 = .95 # correlation between p2 variables
    rho_3 = 0 #.99 # correlation between p3 variables
    
    beta1 = .5 # coef. of z1=x1.1+...+x1.p1
    beta2 = 1  # coef. of z2=x2.1+...+x2.p2
    beta3 = .1 # coef. of variables in X3
    
    # Generating the p1 variables
    X1 = np.random.normal(0,sigma_1,size=(N,p1))
    z1 = X1.dot(np.ones(p1))
    
    # Generating the p2 variables
    Sigma_2 = rho_2 * np.ones((p2,p2)) + (1-rho_2)*np.identity(p2)
    eig_Sigma_2 = np.linalg.eigh(Sigma_2)
    eig_val = eig_Sigma_2[0]
    eig_vec = eig_Sigma_2[1]
    sqrt_Sigma_2 = eig_vec.dot( np.diagflat(np.sqrt(eig_val))).dot(eig_vec.T)

    X2 = np.random.normal(0,sigma_2,size=(N,p2)).dot(sqrt_Sigma_2)
    z2 = X2.dot(np.ones(p2))
     
    # Generating the p3 variables
    Sigma_3 = rho_3 * np.ones((p3,p3)) + (1-rho_3)*np.identity(p3)
    eig_Sigma_3 = np.linalg.eigh(Sigma_3)
    eig_val = eig_Sigma_3[0]
    eig_vec = eig_Sigma_3[1]
    sqrt_Sigma_3 = eig_vec.dot( np.diagflat(np.sqrt(eig_val))).dot(eig_vec.T)

    X3 = np.random.normal(0,sigma_3,size=(N,p3)).dot(sqrt_Sigma_3)
    z3 = X3.dot(np.ones(p3))
    
    X = np.concatenate((X1,X2,X3),axis=1)
    
    # defining the response variable
    y = beta1*z1 + beta2*z2 + beta3*z3 + np.random.normal(0,sigma_eps,size=(N))

    return X, y

def lm_OLS_function(X, y, newX=False, fit_cte=True):
    N, P = np.shape(X)
    newX_yes = (type(newX)==type(X))
    if newX_yes: newN, newP = np.shape(newX)
    if fit_cte:
        X = np.concatenate((X,np.ones((N,1))),axis=1)
        if newX_yes: newX = np.concatenate((newX,np.ones((newN,1))),axis=1)
        
    beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    y_hat = X.dot(beta_hat)
    if newX_yes: 
        newy_hat = newX.dot(beta_hat)
        return beta_hat, y_hat, newy_hat
    else:
        return beta_hat, y_hat
    
class lm_OLS():
    def __init__(self,*,fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        N, P = np.shape(X)
        if self.fit_intercept:
            X = np.concatenate((X,np.ones((N,1))),axis=1)
        self.beta_hat_ = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        return self

    def fit_y(self, X):
        self.y_hat_ = self.predict(X)
        return self

    def predict(self, newX):
        newN, newP = np.shape(newX)
        if self.fit_intercept:
            newX = np.concatenate((newX,np.ones((newN,1))),axis=1)
        return newX.dot(self.beta_hat_)
    
def fit_lm_OLS(X, y, fit_intercept=True): # modified from pyhrt
    lm_ols = lm_OLS(fit_intercept=fit_intercept)
    lm_ols.fit(X, y)
#    lm_ols.fit_y(X)
    return lm_ols

class OLS: # from pyhrt
    def __init__(self):
        pass

    def fit(self, X, y):
        self.coef_ = np.linalg.solve(X.T.dot(X), X.T.dot(y))

    def predict(self, X):
        return X.dot(self.coef_)

def fit_ols(X, y): # from pyhrt
    ols = OLS()
    ols.fit(X, y)
    return ols

def fit_lasso(X, y, cv=5, n_alphas=10, max_iter=250): # modified from pyhrt
    ''' Fit a lasso model ''' # n_alphas=10, max_iter=250 are Ok for model_j
    lasso = LassoCV(cv=cv, n_alphas=n_alphas, max_iter=max_iter,selection='random')
    lasso.fit(X, y)
    return lasso


# Modified from basic_hrt() in timings.py, from pyhrt
def my_basic_hrt(X_test,y_test, model, verbose_level=1, nepochs=20, val_pct=0.1, ntrials=10000):#, **kwargs):
    Nt, P = np.shape(X_test)
    # Get the test MSE
    pred = model.predict(X_test)
    t_true = ((y_test - pred)**2).sum()

    # Run the basic HRT that avoids refitting on the test set
    rel_hrt = np.zeros(P)
    p_value = np.zeros(P)
    for j in range(P):
        if (verbose_level>=1):
            print('\t\tFeature {} of {}'.format(j+1,P))
            sys.stdout.flush()
            
        X_test_j = np.copy(X_test)
        # Sample a null column for the target feature
        xj = X_test[:,j]
        X_test_no_j = np.delete(Xt,j,1)
        mdn_j = fit_mdn(X_test_no_j, xj, verbose=(verbose_level>=2), nepochs=nepochs, val_pct=val_pct)
        fitted_mdn_j=mdn_j.predict(X_test_no_j)
        for trial in range(ntrials):
            X_test_j[:,j] = fitted_mdn_j.sample()
            # Get the null test MSE
            pred = model.predict(X_test_j)
            t_null = ((y_test - pred)**2).sum()

            # Cumulate the values of t_null as a variable relevance measure
            rel_hrt[j] += t_null
            
            # Add 1 if the null was at least as good as the true feature
            p_value[j] += int(t_true >= t_null)

    # Return the one-sided p-value
    return (rel_hrt/ntrials)/t_true, (1+p_value)/(1+ntrials), 

class relev_loco():
    def __init__(
        self, 
        *, 
        model, 
    ):
        self.model=model
        
    def compute_rel_loco(self, X, y, Xt, yt):
        # Predicting in the training sample 
        N, P = np.shape(X)
        self.N = N
        y_hat = self.model.predict(X)
        self.MSPE = (y-y_hat).T.dot(y-y_hat)/N

        # Predicting in the test sample 
        Nt, P = np.shape(Xt)
        self.Nt = Nt
        yt_hat = self.model.predict(Xt)
        self.MSPEt = (yt-yt_hat).T.dot(yt-yt_hat)/Nt

        # training sample
        # measures using (y_hat-y_hat_no_j)^2
        self.relev_loco_tr = np.zeros(P)
        # measures using (y-y_hat_no_j)^2
        self.relev_loco_tr_e = np.zeros(P)

        # test sample
        # measures using (yt_hat-yt_hat_no_j)^2
        self.relev_loco_ts = np.zeros(P)
        # measures using (yt-yt_hat_no_j)^2
        self.relev_loco_ts_e = np.zeros(P)
        
        for j in range(P):
            X_no_j = np.delete(X,j,1)
            Xt_no_j = np.delete(Xt,j,1)
            model_no_j = type(self.model)()
            model_no_j.fit(X_no_j,y)
            y_hat_no_j = model_no_j.predict(X_no_j)
            yt_hat_no_j = model_no_j.predict(Xt_no_j)

            self.relev_loco_tr[j]   = ((y_hat - y_hat_no_j)**2).mean()/ self.MSPE
            self.relev_loco_tr_e[j] = ((y - y_hat_no_j)**2).mean()/ self.MSPE
            self.relev_loco_ts[j]   = ((yt_hat - yt_hat_no_j)**2).mean()/ self.MSPEt
            self.relev_loco_ts_e[j] = ((yt - yt_hat_no_j)**2).mean()/ self.MSPEt
        return self


class relev_ghost_var():
    def __init__(
        self, 
        *, 
        model, 
    ):
        self.model=model
        
    def compute_rel_gh(self, Xt, yt, model_j=fit_lm_OLS):
        Nt, P = np.shape(Xt)
        self.Nt = Nt

        # Predicting in the test sample 
        yt_hat = self.model.predict(Xt)
        self.MSPEt = (yt-yt_hat).T.dot(yt-yt_hat)/Nt

        # Matrix of ghost variables        
        self.GhostX = np.zeros((Nt,P))

        # measures using (yt_hat-yt_hat_j)^2
        self.A = np.zeros((Nt,P))
        self.relev_ghost = np.zeros(P)

        # *_e: measures using (yt-yt_hat_j)^2
        self.A_e = np.zeros((Nt,P))
        self.relev_ghost_e = np.zeros(P)
        
        for j in range(P):
            xj = Xt[:,j]
            Xt_no_j = np.delete(Xt,j,1)
            #reg_j = model_j()
            #reg_j.fit(Xt_no_j, xj)   
            reg_j = model_j(Xt_no_j, xj)
            xj_hat = reg_j.predict(Xt_no_j)
    
            Xt_j = np.copy(Xt)
            Xt_j[:,j] = xj_hat
            yt_hat_j = self.model.predict(Xt_j)

            self.GhostX[:,j] = xj_hat
            self.A[:,j] = yt_hat - yt_hat_j
            self.relev_ghost[j] = (self.A[:,j]**2).mean()/ self.MSPEt
            self.A_e[:,j] = yt - yt_hat_j
            self.relev_ghost_e[j] = (self.A_e[:,j]**2).mean()/ self.MSPEt
        return self

    def V(self):
        self.V = (1/self.Nt)* self.A.T.dot(self.A) / self.MSPEt
        return self
   
    def eig_V(self):
        eig_V = np.linalg.eigh(self.V)
        self.eig_val = eig_V[0]
        self.eig_vec = eig_V[1]
        return self

    def V_e(self):
        self.V_e = (1/self.Nt)* self.A_e.T.dot(self.A_e) / self.MSPEt
        return self
   
    def eig_V_e(self):
        eig_V_e = np.linalg.eigh(self.V_e)
        self.eig_val = eig_V_e[0]
        self.eig_vec = eig_V_e[1]
        return self


if __name__ == '__main__':

    gener_data_model="Hooker" # "2022"
    
    if (gener_data_model=="Hooker"):
        # For Hooker generaring model 
        # How many replicates of the simulation
        runs = 1##100
        # Sample sizes
        N = 2000#2000
        Nt = 1000#1000
        r=0.9
        sP=10
    else:
    # For 2022 generating model
        # How many replicates of the simulation
        runs = 1##100
        # Sample sizes
        N = 2000#2000
        Nt = 1000#1000
        P = [5,45,50]
        #P = [5,5,5]
        sP = sum(P) 


    # reproducibility
    #np.random.seed(123456) 

    relev_loco_tr = np.zeros(sP)
    relev_loco_tr_e = np.zeros(sP)
    relev_loco_ts = np.zeros(sP)
    relev_loco_ts_e = np.zeros(sP)
    time_loco = 0
    relev_ghost = np.zeros(sP)
    relev_ghost_e = np.zeros(sP)
    time_GhVar = 0
    p_val_hrt = np.zeros(sP)
    rel_hrt = np.zeros(sP)
    time_hrt = 0

    for run in range(runs):
        print('Trial {} of {}'.format(run+1,runs))
        if (gener_data_model=="Hooker"):
            X, y = generating_model_Hooker(N, r)
            Xt, yt = generating_model_Hooker(Nt, r)
        else:
            X, y = generating_model_2022(N, P)
            Xt, yt = generating_model_2022(Nt, P)
        
        

        # Fit a lm using least squares with our own function
        # Using classes:
        #reg_Xy =  fit_lm_OLS(X,y)
        reg_Xy =  fit_lasso(X, y, cv=5, n_alphas=20, max_iter=500)

        yt_hat = reg_Xy.predict(Xt)
        
        # Compute the variables relevance by LOCO
        start = time.time()
        rel_loco = relev_loco(model=reg_Xy).compute_rel_loco(X,y,Xt,yt)
        relev_loco_tr += rel_loco.relev_loco_tr
        relev_loco_tr_e += rel_loco.relev_loco_tr_e
        relev_loco_ts += rel_loco.relev_loco_ts
        relev_loco_ts_e += rel_loco.relev_loco_ts_e
        end = time.time()
        time_loco += (end-start)
        
        # Compute the variables relevance by Ghost Variables
        start = time.time()
        rel_GhVar = relev_ghost_var(model=reg_Xy).compute_rel_gh(Xt,yt)
        #rel_GhVar \
        #    = relev_ghost_var(model=reg_Xy).compute_rel_gh(Xt,yt,model_j=fit_lasso)
        relev_ghost += rel_GhVar.relev_ghost
        relev_ghost_e += rel_GhVar.relev_ghost_e
        end = time.time()
        time_GhVar += (end-start)
        
        # hrt 
        if (1==0):
            start = time.time()
            ntrials = 100
            nepochs = 20
            out_hrt = my_basic_hrt(Xt, yt, model=reg_Xy, verbose_level=1, nepochs=nepochs, val_pct=0.2, ntrials=ntrials)
            rel_hrt += out_hrt[0]
            p_val_hrt += out_hrt[1]
            end = time.time()
            time_hrt += (end-start)
    
    print("Time loco={}. Time RelGhVar={}. Time hrt={}".format(time_loco, time_GhVar, time_hrt))

    relev_loco_tr = relev_loco_tr/runs
    relev_loco_tr_e = relev_loco_tr_e/runs
    relev_loco_ts = relev_loco_ts/runs
    relev_loco_ts_e = relev_loco_ts_e/runs

    relev_ghost = relev_ghost/runs
    relev_ghost_e = relev_ghost_e/runs

    rel_hrt = rel_hrt/runs
    p_val_hrt = p_val_hrt/runs

    fig1, axs = plt.subplots(nrows=2, ncols=2)
    axs[0,0].plot(relev_loco_ts,color="red")
    axs[0,0].plot(relev_ghost)
    axs[0,0].set_title("Ghost Variables")
    axs[0,0].set_xlabel("Features")
    axs[0,0].set_ylabel("Relevance")

    axs[0,1].plot(relev_loco_ts_e,color="red")
    axs[0,1].plot(relev_ghost_e)
    axs[0,1].set_title("Ghost Variables (to y_test)")
    axs[0,1].set_xlabel("Features")
    axs[0,1].set_ylabel("Relevance")

    axs[1,0].plot(relev_loco_ts_e,color="red")
    axs[1,0].plot(rel_hrt)
    axs[1,0].set_title("Relevance by hrt")
    axs[1,0].set_xlabel("Features")
    axs[1,0].set_ylabel("Relevance")

    axs[1,1].plot(p_val_hrt)
    axs[1,1].set_title("p-values by hrt")
    axs[1,1].set_xlabel("Features")
    axs[1,1].set_ylabel("p-values")


                


