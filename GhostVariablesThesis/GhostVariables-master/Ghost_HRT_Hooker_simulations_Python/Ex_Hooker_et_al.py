# -*- coding: utf-8 -*-
'''
Created on May 6 2022
@author: delicado

Relevance by Ghost Variables.
Code arranged from 'Ex_GhVar.py' and 'aux_functs_simul_Ex_Hooker_et_al.R'
'''

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm
#from sklearn.linear_model import LinearRegression
#from collections import defaultdict
#from pyhrt.utils import p_value_2sided
#from sklearn import linear_model
from tools_from_pyhrt.continuous import fit_mdn#, GaussianMixtureModel, MixtureDensityNetwork
#from utils.iofuncs import save_dictionary
#from utils.iofuncs import load_dictionary

def generating_model_Hooker(N, r=0.9, s_eps=.5):
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
    y = ytrue + s_eps*np.random.normal(size=(N,))
    return X, y

def generating_model_Hooker_X(N, r=0.9):
    #### from Hooker et al :   
    #n1 <- 2000 # size of the training sample 
    #n2 <- 1000 # size of the test sample
    #nsim = 50
    nfeat  = 10
    # generating the explanatory variables
    X = np.random.uniform(size=(N,nfeat))
    R = np.array([[1, r], [0, np.sqrt(1-r**2)]])
    X[:,:2] = norm.cdf(norm.ppf(X[:,:2]).dot(R))
    return X

def generating_model_Hooker_y(X, s_eps=.5):
    N = np.shape(X)[0]
    ytrue = 1*X[:,0] + 1*X[:,1] + X[:,2:5].sum(axis=1) + \
         0.5*X[:,6] + 0.8*X[:,7] + 1.2*X[:,8] + 1.5*X[:,9] 
    y = ytrue + s_eps*np.random.normal(size=(N,))
    return y

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
    return (rel_hrt/ntrials)/t_true, (1+p_value)/(1+ntrials)

class relev_loco():
    def __init__(self, *, model):
        self.model =model
                
    def compute_rel_loco(self, X, y, Xt, yt, MSPEt, yt_hat):
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

        # test sample
        # measures using (yt_hat-yt_hat_no_j)^2
        self.rel_loco = np.zeros(P)
        # measures using (yt-yt_hat_no_j)^2
        self.rel_loco_e = np.zeros(P)
        
        for j in range(P):
            X_no_j = np.delete(X,j,1)
            Xt_no_j = np.delete(Xt,j,1)
            model_no_j = type(self.model)()
            model_no_j.fit(X_no_j,y)
            #y_hat_no_j = model_no_j.predict(X_no_j)
            yt_hat_no_j = model_no_j.predict(Xt_no_j)

            #self.relev_loco_tr[j]   = ((y_hat - y_hat_no_j)**2).mean()/ self.MSPE
            #self.relev_loco_tr_e[j] = ((y - y_hat_no_j)**2).mean()/ self.MSPE
            self.rel_loco[j]   = ((yt_hat - yt_hat_no_j)**2).mean()/ MSPEt
            self.rel_loco_e[j] = ((yt - yt_hat_no_j)**2).mean()/ MSPEt - 1
        return self


class relev_ghost_var():
    def __init__(self):
        self.aux=0
        
    def compute_rel_gh(self, Xt, yt, model, model_j=fit_lm_OLS):
        Nt, P = np.shape(Xt)
        self.Nt = Nt

        # Predicting in the test sample 
        yt_hat = model.predict(Xt)
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
            yt_hat_j = model.predict(Xt_j)

            self.GhostX[:,j] = xj_hat
            self.A[:,j] = yt_hat - yt_hat_j
            self.relev_ghost[j] = (self.A[:,j]**2).mean()/ self.MSPEt
            self.A_e[:,j] = yt - yt_hat_j
            self.relev_ghost_e[j] = (self.A_e[:,j]**2).mean()/ self.MSPEt - 1
        return self

    # new code May 6 2022:
    def create_GhostX(self, Xt, model_j=fit_lm_OLS):
        Nt, P = np.shape(Xt)
        # Matrix of ghost variables:
        self.GhostX = np.zeros((Nt,P))
        for j in range(P):
            xj = Xt[:,j]
            Xt_no_j = np.delete(Xt,j,1)
            reg_j = model_j(Xt_no_j, xj)
            xj_hat = reg_j.predict(Xt_no_j)
            self.GhostX[:,j] = xj_hat
        return self
    
    def compute_rel_gh_from_GhostX(self, Xt, yt, Xt_gh, MSPEt, yt_hat, model):
        Nt, P = np.shape(Xt)
        self.Nt = Nt

        # measures using (yt_hat-yt_hat_j)^2
        self.A = np.zeros((Nt,P))
        self.relev_ghost = np.zeros(P)

        # *_e: measures using (yt-yt_hat_j)^2
        self.A_e = np.zeros((Nt,P))
        self.relev_ghost_e = np.zeros(P)
        
        for j in range(P):
            Xt_j = np.copy(Xt)
            Xt_j[:,j] = Xt_gh[:,j]
            yt_hat_j = model.predict(Xt_j)

            self.A[:,j] = yt_hat - yt_hat_j
            self.relev_ghost[j] = (self.A[:,j]**2).mean()/ MSPEt
            self.A_e[:,j] = yt - yt_hat_j
            self.relev_ghost_e[j] = (self.A_e[:,j]**2).mean()/ MSPEt - 1
        return self

    # end new code May 6 2022

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

# New code, May 6 2022:
# translated from R code
def Vimp_perturb_pred_err(X_test,y_test, model, pert_X_test, MSPE_test, y_test_hat):
    # pert_X_test: Matrix with the "perturbed" columns of Xt
    Nt, P = np.shape(X_test)
    VI_pred = np.zeros(P)
    VI_err  = np.zeros(P)
    for j in range(P):
        Xp = np.copy(X_test)
        Xp[:,j] = pert_X_test[:,j]
        yt_hat_j = model.predict(Xp)
        # relative VI based on predictions:
        VI_pred[j] = ((y_test_hat - yt_hat_j)**2).mean()/MSPE_test    
        # relative VI based on predictiona errors:
        VI_err[j]  = ((y_test     - yt_hat_j)**2).mean()/MSPE_test -1 
    return VI_pred, VI_err

def perturb_hrt_mdn(X, nepochs=20, val_pct=0.1):#, **kwargs):
    N, P = np.shape(X)
    X_mdn = np.copy(X)
    for j in range(P):          
        # Sample a null column for the target feature
        xj = X[:,j]
        X_no_j = np.delete(X,j,1)
        mdn_j = fit_mdn(X_no_j, xj, verbose=False, nepochs=nepochs, val_pct=val_pct)
        xj_mdn=mdn_j.predict(X_no_j)
        X_mdn[:,j] = xj_mdn.sample()
    return X_mdn

# end New code, May 6 2022


if __name__ == '__main__':

    gener_data_model="Hooker" # May 6 2022
    
    # For Hooker generaring model 
    # How many replicates of the simulation
    runs = 50##50    # nsim, number of simulations
    # Sample sizes
    N = 2000#2000     # training set size N
    Nt = 1000#1000    # test size
    
    rs=[0,0.9] # vector of rho's=cor(X_1,X_2)
    s_eps=.1   # std. dev. of the noise epsilon
    sP=10 # nfeat, number of variables

    use_loco = True # False
    
    # reproducibility
    np.random.seed(123456) 

    dims = [runs,len(rs),sP]
    if (use_loco):
        rel_loco = np.zeros(shape=dims)
        rel_loco_e = np.zeros(shape=dims)
        rel_loco_rank = np.zeros(shape=dims)
        rel_loco_e_rank = np.zeros(shape=dims)
        time_loco = 0

    relev_ghost = np.zeros(shape=dims)
    relev_ghost_e = np.zeros(shape=dims)
    relev_ghost_rank = np.zeros(shape=dims)
    relev_ghost_e_rank = np.zeros(shape=dims)
    time_GhVar = 0

    #p_val_hrt = np.zeros(shape=dims)
    relev_hrt = np.zeros(shape=dims)
    relev_hrt_e = np.zeros(shape=dims)
    relev_hrt_rank = np.zeros(shape=dims)
    relev_hrt_e_rank = np.zeros(shape=dims)
    time_hrt = 0

    time_model = 0
    
    ind_r = 0 
    for run in range(runs):
        print('Trial {} of {}'.format(run+1,runs))
        ind_r = 0
        for r in rs:
            print('   rho= {}'.format(r))
            X  = generating_model_Hooker_X(N, r)
            Xt = generating_model_Hooker_X(Nt,r)
            y  = generating_model_Hooker_y(X, s_eps=.5)
            yt = generating_model_Hooker_y(Xt,s_eps=.5)

            # Creating the perturbed variable matrices
            start = time.time()
            X_gh = relev_ghost_var().create_GhostX(X, model_j=fit_lm_OLS).GhostX
            Xt_gh = relev_ghost_var().create_GhostX(Xt, model_j=fit_lm_OLS).GhostX
            end = time.time()
            time_GhVar += (end-start)
            print('Partial time_GhVar= {}'.format(time_GhVar))
            
            start = time.time()
            X_mdn  = perturb_hrt_mdn(X, nepochs=20, val_pct=0.1)
            Xt_mdn = perturb_hrt_mdn(Xt,nepochs=20, val_pct=0.1)
            end = time.time()
            time_hrt += (end-start)
            print('Partial time_hrt= {}'.format(time_hrt))
             
            # Initialize the fitting model
            
            # Linear model estimated by OLS:
            # Fit a lm using least squares with our own function
            # Using classes:
            reg_Xy =  fit_lm_OLS(X,y)
            
            # Linear model estimated by Lasso:
            #reg_Xy =  fit_lasso(X, y, cv=5, n_alphas=20, max_iter=500)
            
            # Random Forest
            #reg_Xy = RandomForestRegressor()# n_estimators=100 (it is 500 in R)
            #reg_Xy.fit(X,y)
            
            yt_hat = reg_Xy.predict(Xt)
            MSPEt = (yt-yt_hat).T.dot(yt-yt_hat)/Nt
            
            # Compute the variables relevance by Ghost Variables
            start = time.time()
            # Three equyivalent ways of computing rel_GhVar:
            #rel_GhVar \
            #    = relev_ghost_var(model=reg_Xy).compute_rel_gh(Xt,yt,model_j=fit_lasso)
            #rel_GhVar \
            #    = relev_ghost_var(model=reg_Xy).compute_rel_gh_from_GhostX(Xt,yt,Xt_gh,MSPEt, yt_hat)
            #relev_ghost += rel_GhVar.relev_ghost
            #relev_ghost_e += rel_GhVar.relev_ghost_e
            rel_GhVar \
                = Vimp_perturb_pred_err(Xt,yt, reg_Xy, Xt_gh, MSPEt, yt_hat)
            relev_ghost[run,ind_r,:] = rel_GhVar[0]
            relev_ghost_e[run,ind_r,:] = rel_GhVar[1]
            end = time.time()
            time_GhVar += (end-start)
            relev_ghost_rank[run,ind_r,:] = relev_ghost[run,ind_r,:].argsort().argsort()
            relev_ghost_e_rank[run,ind_r,:] = relev_ghost_e[run,ind_r,:].argsort().argsort()
            
            # hrt 
            start = time.time()
            #out_hrt = my_basic_hrt(Xt, yt, model=reg_Xy, verbose_level=1, nepochs=nepochs, val_pct=0.2, ntrials=ntrials)
            #rel_hrt += out_hrt[0]
            #p_val_hrt += out_hrt[1]
            out_hrt = Vimp_perturb_pred_err(Xt,yt, reg_Xy, Xt_mdn, MSPEt, yt_hat)
            relev_hrt[run,ind_r,:] = out_hrt[0]
            relev_hrt_e[run,ind_r,:] = out_hrt[1]
            end = time.time()
            time_hrt += (end-start)
            relev_hrt_rank[run,ind_r,:] = relev_hrt[run,ind_r,:].argsort().argsort()
            relev_hrt_e_rank[run,ind_r,:] = relev_hrt_e[run,ind_r,:].argsort().argsort()

            # Compute the variables relevance by LOCO
            if (use_loco):
                start = time.time()
                rel_loco_test = relev_loco(model=reg_Xy).compute_rel_loco(X,y,Xt,yt, MSPEt, yt_hat)
                rel_loco[run,ind_r,:] = rel_loco_test.rel_loco
                rel_loco_e[run,ind_r,:] = rel_loco_test.rel_loco_e
                end = time.time()
                time_loco += (end-start)
                print('Partial time_loco= {}'.format(time_loco))
                rel_loco_rank[run,ind_r,:] = rel_loco[run,ind_r,:].argsort().argsort()
                rel_loco_e_rank[run,ind_r,:] = rel_loco_e[run,ind_r,:].argsort().argsort()

            ind_r += 1
    
    print("Time RelGhVar={}. Time hrt={}".format(time_GhVar, time_hrt))
    if (use_loco):
        print("Time loco={}".format(time_loco))
        
    #save_dictionary(globals(),'output_simulation/simul_2022_05_09.spydata')
    #load_dictionary('output_simulation/simul_2022_05_09.spydata')  
    # np.savez('output_simulation/simul_2022_05_10.txt',\ # random forest
    # np.savez('output_simulation/simul_2022_05_17_lm_OLS.txt',\ # linear model by OLS
    np.savez('output_simulation/simul_2022_05_17_lm_OLS.txt',\
             relev_ghost,relev_ghost_e,relev_hrt,relev_hrt_e,\
             relev_ghost_rank,relev_ghost_e_rank,relev_hrt_rank,relev_hrt_e_rank,\
             rel_loco,rel_loco_e,\
             rel_loco_rank,rel_loco_e_rank,\
             time_GhVar,time_hrt,time_loco, time_model,
             delimeter=";"\
            )

    fig1, axs = plt.subplots(nrows=2, ncols=2)
    if (use_loco):
        axs[0,0].plot(rel_loco[:,0,].mean(axis=0),color="green")
    axs[0,0].plot(relev_ghost[:,0,].mean(axis=0),color="blue")
    axs[0,0].plot(relev_hrt[:,0,].mean(axis=0),color="red")
    axs[0,0].set_title("Rho=0. Relevance by diffs in predictions")
    axs[0,0].set_xlabel("Features")
    axs[0,0].set_ylabel("Relevance")

    if (use_loco):
        axs[0,1].plot(rel_loco_e[:,0,].mean(axis=0),color="green")
    axs[0,1].plot(relev_ghost_e[:,0,].mean(axis=0),color="blue")
    axs[0,1].plot(relev_hrt_e[:,0,].mean(axis=0),color="red")
    axs[0,1].set_title("Rho=0. Relevance by diffs in MSPE")
    axs[0,1].set_xlabel("Features")
    axs[0,1].set_ylabel("Relevance")

    if (use_loco):
        axs[1,0].plot(rel_loco[:,1,].mean(axis=0),color="green")
    axs[1,0].plot(relev_ghost[:,1,].mean(axis=0),color="blue")
    axs[1,0].plot(relev_hrt[:,1,].mean(axis=0),color="red")
    axs[1,0].set_title("Rho=0.9. Relevance by diffs in predictions")
    axs[1,0].set_xlabel("Features")
    axs[1,0].set_ylabel("Relevance")

    if (use_loco):
        axs[1,1].plot(rel_loco_e[:,1,].mean(axis=0),color="green")
    axs[1,1].plot(relev_ghost_e[:,1,].mean(axis=0),color="blue")
    axs[1,1].plot(relev_hrt_e[:,1,].mean(axis=0),color="red")
    axs[1,1].set_title("Rho=0.9. Relevance by diffs in MSPE")
    axs[1,1].set_xlabel("Features")
    axs[1,1].set_ylabel("Relevance")

    fig1, axs = plt.subplots(nrows=2, ncols=2)
    if (use_loco):
        axs[0,0].plot(rel_loco_rank[:,0,].mean(axis=0),color="green")
    axs[0,0].plot(relev_ghost_rank[:,0,].mean(axis=0),color="blue")
    axs[0,0].plot(relev_hrt_rank[:,0,].mean(axis=0),color="red")
    axs[0,0].set_title("Rho=0. Rank relevance by diffs in predictions")
    axs[0,0].set_xlabel("Features")
    axs[0,0].set_ylabel("Relevance")

    if (use_loco):
        axs[0,1].plot(rel_loco_e_rank[:,0,].mean(axis=0),color="green")
    axs[0,1].plot(relev_ghost_e_rank[:,0,].mean(axis=0),color="blue")
    axs[0,1].plot(relev_hrt_e_rank[:,0,].mean(axis=0),color="red")
    axs[0,1].set_title("Rho=0. Rank relevance by diffs in MSPE")
    axs[0,1].set_xlabel("Features")
    axs[0,1].set_ylabel("Relevance")

    if (use_loco):
        axs[1,0].plot(rel_loco_rank[:,1,].mean(axis=0),color="green")
    axs[1,0].plot(relev_ghost_rank[:,1,].mean(axis=0),color="blue")
    axs[1,0].plot(relev_hrt_rank[:,1,].mean(axis=0),color="red")
    axs[1,0].set_title("Rho=0.9. Rank relevance by diffs in predictions")
    axs[1,0].set_xlabel("Features")
    axs[1,0].set_ylabel("Relevance")

    if (use_loco):
        axs[1,1].plot(rel_loco_e_rank[:,1,].mean(axis=0),color="green")
    axs[1,1].plot(relev_ghost_e_rank[:,1,].mean(axis=0),color="blue")
    axs[1,1].plot(relev_hrt_e_rank[:,1,].mean(axis=0),color="red")
    axs[1,1].set_title("Rho=0.9. Rank relevance by diffs in MSPE")
    axs[1,1].set_xlabel("Features")
    axs[1,1].set_ylabel("Relevance")

