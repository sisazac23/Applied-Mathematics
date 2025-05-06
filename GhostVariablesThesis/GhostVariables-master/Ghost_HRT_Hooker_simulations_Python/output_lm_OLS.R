#install.packages("reticulate")
library(reticulate)
np <- import("numpy")

npzfile <- np$load("output_simulation/simul_2022_05_17_lm_OLS.txt.npz")

npzfile$files
# [1] "delimeter" "arr_0"     "arr_1"     "arr_2"     "arr_3"    
# [6] "arr_4"     "arr_5"     "arr_6"     "arr_7"     "arr_8"    
# [11] "arr_9"     "arr_10"    "arr_11"    "arr_12"    "arr_13"   
# [16] "arr_14"    "arr_15"

relev_ghost=npzfile$f[['arr_0']]
relev_ghost_e=npzfile$f[['arr_1']]
relev_hrt=npzfile$f[['arr_2']]
relev_hrt_e=npzfile$f[['arr_3']]
relev_ghost_rank=npzfile$f[['arr_4']]
relev_ghost_e_rank=npzfile$f[['arr_5']]
relev_hrt_rank=npzfile$f[['arr_6']]
relev_hrt_e_rank=npzfile$f[['arr_7']]
rel_loco=npzfile$f[['arr_8']]
rel_loco_e=npzfile$f[['arr_9']]
rel_loco_rank=npzfile$f[['arr_10']]
rel_loco_e_rank=npzfile$f[['arr_11']]
time_GhVar=npzfile$f[['arr_12']]
time_hrt=npzfile$f[['arr_13']]
time_loco=npzfile$f[['arr_14']] 
time_model=npzfile$f[['arr_15']]

print(paste("Time RelGhVar=",time_GhVar))
print(paste("Time hrt=",time_hrt))
print(paste("Time loco=",time_loco))

Imps.hrt.lin.m  = apply(relev_ghost_rank+1,c(2,3),mean)
Imps.hrt.lin.s  = apply(relev_ghost_rank+1,c(2,3),sd)
Imps.hrt.lin.e.m  = apply(relev_ghost_e_rank+1,c(2,3),mean)
Imps.hrt.lin.e.s  = apply(relev_ghost_e_rank+1,c(2,3),sd)
Imps.loco.lin.m = apply(rel_loco_rank+1,c(2,3),mean)
Imps.loco.lin.s = apply(rel_loco_rank+1,c(2,3),sd)
Imps.loco.lin.e.m = apply(rel_loco_e_rank+1,c(2,3),mean)
Imps.loco.lin.e.s = apply(rel_loco_e_rank+1,c(2,3),sd)
rel.gh.lin.m    = apply(relev_ghost_rank+1,c(2,3),mean)
rel.gh.lin.s    = apply(relev_ghost_rank+1,c(2,3),sd)
rel.gh.lin.e.m  = apply(relev_ghost_e_rank+1,c(2,3),mean)
rel.gh.lin.e.s  = apply(relev_ghost_e_rank+1,c(2,3),sd)

rs <- c(0,.9)
nfeat <- 10 

printing = FALSE#TRUE
if (printing) pdf("output_lm_OLS_Python.pdf", width = 10, height=5)
op <- par(mfrow=c(1,2))
for(k in 1:length(rs)){
  par(mar=c(5,5,4,1))
  plot(Imps.hrt.lin.e.m[k,],type='b',col=2,pch=2,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.loco.lin.m[k,],type='b',col=2,pch=19,cex=1.5,lwd=2)
  lines(Imps.loco.lin.e.m[k,],type='b',col=3,pch=3,cex=1.5,lwd=2)
  #lines(Imps.loco.lin.m[k,],type='b',col=3,pch=19,cex=1.5,lwd=2)
  lines(rel.gh.lin.e.m[k,],type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lin.m[k,],type='b',col=5,pch=5,cex=1.5,lwd=2)
  #legend('bottomright',c('eC','eCe','L','Le','G','Ge'), pch=c(19,2,19,3:5),col=c(2,2,3,3:5),cex=1.5,lwd=2)
  legend('bottomright',c('eC','L','G','Gp'), pch=c(2,3:5),col=c(2,3:5),cex=1,lwd=2)  
  title(bquote('Linear model fitted by OLS.'~rho==.(rs[k])))
}
par(op)
if (printing) dev.off()

# rho = 0
plot(rel_loco[,1,],relev_hrt[,1,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(rel_loco[,1,],relev_ghost[,1,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(rel_loco_e[,1,],relev_hrt_e[,1,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(rel_loco_e[,1,],relev_ghost_e[,1,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(relev_ghost[,1,],relev_ghost_e[,1,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

# rho = 0.9
plot(rel_loco[,2,],relev_hrt[,2,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(rel_loco[,2,],relev_ghost[,2,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(rel_loco_e[,2,],relev_hrt_e[,2,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(rel_loco_e[,2,],relev_ghost_e[,2,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

plot(relev_ghost[,2,],relev_ghost_e[,2,],
     col=matrix(1:10,byrow=T,ncol=10,nrow=50));abline(a=0,b=1,col=2)

