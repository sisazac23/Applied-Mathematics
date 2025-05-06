# -*- coding: utf-8 -*-
"""
Created on Wed May 11 13:31:18 2022

@author: delicado
"""
import numpy as np
import matplotlib.pyplot as plt

npzfile = np.load('output_simulation/simul_2022_05_10.txt.npz')

relev_ghost=npzfile['arr_0']
relev_ghost_e=npzfile['arr_1']
relev_hrt=npzfile['arr_2']
relev_hrt_e=npzfile['arr_3']
relev_ghost_rank=npzfile['arr_4']
relev_ghost_e_rank=npzfile['arr_5']
relev_hrt_rank=npzfile['arr_6']
relev_hrt_e_rank=npzfile['arr_7']
rel_loco=npzfile['arr_8']
rel_loco_e=npzfile['arr_9']
rel_loco_rank=npzfile['arr_10']
rel_loco_e_rank=npzfile['arr_11']
time_GhVar=npzfile['arr_12']
time_hrt=npzfile['arr_13']
time_loco=npzfile['arr_14'] 
time_model=npzfile['arr_15']

print("Time RelGhVar={}".format(time_GhVar))
print("Time hrt={}".format(time_hrt))
print("Time loco={}".format(time_loco))

#Time RelGhVar=29.829291343688965
#Time hrt=12771.48453092575
#Time loco=1341.6279697418213

use_loco=True

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

