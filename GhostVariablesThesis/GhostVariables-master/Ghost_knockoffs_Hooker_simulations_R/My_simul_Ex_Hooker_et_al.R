# Recreating the simulated example in Hooker et al. (2021)
source("aux_functs_simul_Ex_Hooker_et_al.R")
#
# Generating model
# (x1,x2)~GaussianCopula(rho), x3...x10 iid U([0,1]), epsilon~N(0,s.eps^2)
# y = x1 + x2 + x3 + x4 + x5 + 0*x6 + 0.5*x7 + 0.8*x8 + 1.2*x9 + 1.5*x10 + epsilon
# 
# Data generating model parameters: 
X12_linear <- TRUE # FALSE # When FALSE, gen_2_vars_not_normal is used
rs = c(0,0.9) # c(.9,0.99) # c(0, .3, .6, .9) #  c(0.3,0.6) # vector of values of rho=cor(X_1,X_2)

s.eps = 0.1 # std. dev. of the noise epsilon
# ratio var(noise)/var(signal):
# s.eps=.1 => ratio=0.01  # Hooker et al simulations
# s.eps=.33=> ratio=0.1
# s.eps=.5 => ratio=0.2

# simulation parameters:
nsim = 50#50   # number of repetitions # Hooker et al simulations: 50
nfeat  = 10 # 10  # number of variables (nnet does not support 25 or more features)
ns = 2000 # 200   # vector of training set sizes n # # Hooker et al simulations: 2000
nts = ns/2  # test set sizes nt

# models to be fitted
fit.lin <- TRUE # FALSE #
fit.rf  <- TRUE # FALSE #
fit.nn  <- TRUE # FALSE #  

model.names <-c("lin","rf","nn")[c(fit.lin,fit.rf,fit.nn)]
time_lin=0
time_rf=0
time_nn=0

# Relevance/Importance methods to be computed
do_loco <- TRUE #  FALSE # 
time.perm <- 0
time.cond <- 0
time.loco <- 0
time.gh <- 0
time.knoc <- 0

##############
if (fit.lin){
  linmod = vector(mode='list',length=nsim)
  Imps.loco.lin = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.perm.lin = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.cond.lin = array(NA,c(length(ns),nsim,length(rs),nfeat))
  rel.gh.lin = array(NA,c(length(ns),nsim,length(rs),nfeat))
  rel.gh.lin.e = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.knoc.lin = array(NA,c(length(ns),nsim,length(rs),nfeat))
  MSPEt.lin = array(NA,c(length(ns),nsim,length(rs)))
}

if (fit.rf){#"Imps.orig" (out-of-bag) from randomForest are not recorded
  rfmod = vector(mode='list',length=nsim)
  Imps.loco.rf = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.perm.rf = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.cond.rf = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.knoc.rf = array(NA,c(length(ns),nsim,length(rs),nfeat))
  rel.gh.rf    = array(NA,c(length(ns),nsim,length(rs),nfeat))
  rel.gh.rf.e  = array(NA,c(length(ns),nsim,length(rs),nfeat))
  MSPEt.rf = array(NA,c(length(ns),nsim,length(rs)))
}

if (fit.nn){
  nnmod = vector(mode='list',length=nsim)
  Imps.loco.nn = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.perm.nn = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.cond.nn = array(NA,c(length(ns),nsim,length(rs),nfeat))
  Imps.knoc.nn = array(NA,c(length(ns),nsim,length(rs),nfeat))
  rel.gh.nn  = array(NA,c(length(ns),nsim,length(rs),nfeat))
  rel.gh.nn.e = array(NA,c(length(ns),nsim,length(rs),nfeat))
  MSPEt.nn = array(NA,c(length(ns),nsim,length(rs)))
}

n = ns[1]
nt = nts[1]

output.file.name = format(Sys.time(), "%Y_%m_%d_%H_%M_%S")

for(j in 1:nsim){
 if (fit.lin) linmod[[j]] = vector(mode='list',length=length(ns))
 if (fit.rf) rfmod[[j]] = vector(mode='list',length=length(ns))
 if (fit.nn) nnmod[[j]] = vector(mode='list',length=length(ns))

 for(i in 1:length(ns)){
  if (fit.lin) linmod[[j]][[i]] = vector(mode='list',length=length(rs)) 
  if (fit.rf) rfmod[[j]][[i]] = vector(mode='list',length=length(rs))
  if (fit.nn) nnmod[[j]][[i]] = vector(mode='list',length=length(rs))
  
  for(k in 1:length(rs)){
    #print(c(i,j,k))
    print(paste("j=",j," of ",nsim,";  ",k, Sys.time()))
    
    r = rs[k]

    # training set
    X = matrix(runif(nfeat*n),n,nfeat) 
    if (X12_linear){
      X[,1:2] = pnorm(qnorm(X[,1:2])%*%matrix(c(1, 0, r, sqrt(1-r^2)),2,2))
    }else{
      X[,1:2] = gen_2_vars_not_normal(n,r)
    }
    # time.cond=time.cond + system.time({
    #   cX = cond.distrib(X,r)
    # })
    # time.knoc=time.knoc + system.time({
    #   kX = knockoff_MX(X)
    # })
    
    ytrue = 1*X[,1] + 1*X[,2] + apply(X[,3:5],1,sum) + 0.5*X[,7] + 0.8*X[,8] + 1.2*X[,9] + 1.5*X[,10] 
    y = ytrue + rnorm(n,sd=s.eps)
    
    # test set
    Xt = matrix(runif(nfeat*nt),nt,nfeat) 
    if (X12_linear){
      Xt[,1:2] = pnorm(qnorm(Xt[,1:2])%*%matrix(c(1, 0, r, sqrt(1-r^2)),2,2))
    }else{
      Xt[,1:2] = gen_2_vars_not_normal(nt,r)
    }
    # Constructing the "perturbed" test sets by
    # random permutations, theoretical conditional distribution,
    # knockoffs, or ghost variables:
    time.perm=time.perm + system.time({
      pXt = random.perm(Xt)    # random permutations
    })
    time.cond=time.cond + system.time({
      cXt = cond.distrib(Xt,r=r,X12_linear = X12_linear) # theoretical conditional distribution
    })
    time.knoc=time.knoc + system.time({
      kXt = knockoff_MX(Xt)    # knockoffs
    })
    time.gh=time.gh + system.time({
      if (X12_linear){
        gXt = ghost_var_lm(Xt)      # ghost variables fitted by lm()
      }else{
        gXt = ghost_var_gam(Xt)     # ghost variables fitted by gam()
      }
    })
    
    yttrue=1*Xt[,1]+1*Xt[,2]+apply(Xt[,3:5],1,sum)+0.5*Xt[,7]+0.8*Xt[,8]+1.2*Xt[,9]+1.5*Xt[,10] 
    yt = yttrue + rnorm(nt,sd=s.eps)
    
    simdat = data.frame(y,X)
  	f = formula(paste('y~',paste(names(simdat)[-1],collapse='+')))
  	
  	# Linear Model
  	if (fit.lin){
  	  ptm <- proc.time()
  	  t.mod = lm(f,data=simdat)
  	  #MSPE = mean( (y - predict(t.mod,data.frame(X)))^2 )
  	  yt.hat <- predict(t.mod,data.frame(Xt))
  	  MSPEt.lin[i,j,k] = mean( (yt - yt.hat)^2 )
  	  linmod[[j]][[i]][[k]] = t.mod
  	  time.perm = time.perm + system.time({
  	    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.lin[i,j,k])
  	    Imps.perm.lin[i,j,k,] = VI.perm
  	  })
  	  time.cond = time.cond + system.time({
  	    VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.lin[i,j,k])
  	    Imps.cond.lin[i,j,k,] = VI.cond
  	  })
  	  time.knoc = time.knoc + system.time({
  	    VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.lin[i,j,k])
  	    Imps.knoc.lin[i,j,k,] = VI.knoc
  	  })
  	  time.gh = time.gh + system.time({
  	    rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.lin[i,j,k],yt.hat)
  	    rel.gh.lin[i,j,k,] <- rel_Gh$relev.ghost
  	    rel.gh.lin.e[i,j,k,] <- rel_Gh$relev.ghost.e
  	  })
  	  time_lin = time_lin + proc.time() - ptm
  	}
  	
  	# Random Forest
  	if (fit.rf){
  	  ptm <- proc.time()
  	  t.mod = randomForest(X,y)
  	  rfmod[[j]][[i]][[k]] = t.mod
  	  #MSPE = mean( (y - predict(t.mod,X))^2 )
  	  yt.hat <- predict(t.mod,Xt)
  	  MSPEt.rf[i,j,k] = mean( (yt - yt.hat)^2 )
  	  time.perm = time.perm + system.time({
  	    VI.perm = Vimp.perturb(t.mod,Xt,yt,pertXt=pXt,MSPEt.rf[i,j,k])
  	    Imps.perm.rf[i,j,k,] = VI.perm
  	  })
  	  time.cond = time.cond + system.time({
  	    VI.cond = Vimp.perturb(t.mod,Xt,yt,pertXt=cXt,MSPEt.rf[i,j,k])
  	    Imps.cond.rf[i,j,k,] = VI.cond
  	  })
  	  time.knoc = time.knoc + system.time({
  	    VI.knoc = Vimp.perturb(t.mod,Xt,yt,pertXt=kXt,MSPEt.rf[i,j,k])
  	    Imps.knoc.rf[i,j,k,] = VI.knoc
  	  })
  	  time.gh = time.gh + system.time({
  	    rel_Gh <- Vimp.perturb.gh(t.mod,Xt,yt,pertXt=gXt,MSPEt.rf[i,j,k],yt.hat)
  	    rel.gh.rf[i,j,k,]   <- rel_Gh$relev.ghost
  	    rel.gh.rf.e[i,j,k,] <- rel_Gh$relev.ghost.e
  	  })
  	  time_rf = time_rf + proc.time() - ptm
  	}
  	
  	# Neural Network
  	if (fit.nn){
  	  ptm <- proc.time()
  	  t.mod = nnetselect(f,data=simdat,ytrue=ytrue, # Attention: ytrue is given as an argument!
  	                     size=2*(nfeat),its=10,linout=TRUE,trace=FALSE)
  	  #	t.mod = nnet(f,data=simdat,size=nfeat,maxit=1000,linout=TRUE,trace=FALSE)
  	  #MSPE = mean( (y - predict(t.mod,data.frame(X)))^2 )
  	  yt.hat <- predict(t.mod,data.frame(Xt))
  	  MSPEt.nn[i,j,k] = mean( (yt - yt.hat)^2 )
  	  nnmod[[j]][[i]][[k]] = t.mod
  	  time.perm = time.perm + system.time({
  	    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.nn[i,j,k])
  	    Imps.perm.nn[i,j,k,] = VI.perm
  	  })
  	  time.cond = time.cond + system.time({
  	    VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.nn[i,j,k])
  	    Imps.cond.nn[i,j,k,] = VI.cond
  	  })
  	  time.knoc = time.knoc + system.time({
  	    VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.nn[i,j,k])
  	    Imps.knoc.nn[i,j,k,] = VI.knoc
  	  })
  	  time.gh = time.gh + system.time({
  	    rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.nn[i,j,k],yt.hat)
  	    rel.gh.nn[i,j,k,] <- rel_Gh$relev.ghost
  	    rel.gh.nn.e[i,j,k,] <- rel_Gh$relev.ghost.e
  	  })
  	  time_nn = time_nn + proc.time() - ptm
  	}
  	
  	# Now versions of re-training models: only loco has been implemented
  	if (do_loco){
  	  ptm.loco <- proc.time()
  	  for(l in 1:ncol(X)){
  	    # Dropping covariates: loco, leave-one-covariate-out
  	    simdat = data.frame(y,X[,-l])
  	    f = formula(paste('y~',paste(names(simdat)[-1],collapse='+')))
  	    if (fit.lin){
  	      ptm <- proc.time()
          t.mod = lm(f,simdat)
          Imps.loco.lin[i,j,k,l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.lin[i,j,k] -1 # relative VI	
          time_lin = time_lin + proc.time() - ptm
  	    }
  	    if (fit.rf){
  	      ptm <- proc.time()
  	      t.mod = randomForest(X[,-l],y)
  	      Imps.loco.rf[i,j,k,l] = mean(  (yt - predict(t.mod,Xt[,-l]))^2 )/MSPEt.rf[i,j,k] -1 # relative VI
  	      time_rf = time_rf + proc.time() - ptm
  	    }
  	    if (fit.nn){
  	      ptm <- proc.time()
  	      t.mod = nnetselect(f,data=simdat,ytrue=ytrue,size=2*(nfeat),its=10,linout=TRUE,trace=FALSE)
  	      #		t.mod = nnet(f,simdat,size=nfeat,maxit=1000,linout=TRUE,trace=FALSE)
  	      Imps.loco.nn[i,j,k,l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.nn[i,j,k] -1 # relative VI
  	      time_nn = time_nn + proc.time() - ptm
  	    }
  	  }
  	  time.loco <- time.loco + proc.time() - ptm.loco
  	}
  }
 }
 times.VI.methods <- data.frame(
   time.perm=time.perm[3],
   time.cond=time.cond[3],
   time.loco=time.loco[3],
   time.gh=time.gh[3],
   time.knoc=time.knoc[3]
 )
 times_models <- data.frame(
   time_lin=time_lin[3],
   time_rf=time_rf[3],
   time_nn=time_nn[3]
 )
 if ((100*(j/nsim))%%100==0) save.image(paste0('output_simulation/',output.file.name,'.Rdata'),safe=TRUE)
}

save.image(paste0('output_simulation/',output.file.name,'.Rdata'),safe=TRUE)

print(times.VI.methods)
print(times_models)

# Now lets look at some of this: means and standard deviations of importances
if (fit.lin){
  Imps.perm.lin.m = apply(apply(Imps.perm.lin[,,,],1:2,rank),c(1,3),mean)
  Imps.perm.lin.s = apply(apply(Imps.perm.lin[,,,],1:2,rank),c(1,3),sd)
  Imps.cond.lin.m = apply(apply(Imps.cond.lin[,,,],1:2,rank),c(1,3),mean)
  Imps.cond.lin.s = apply(apply(Imps.cond.lin[,,,],1:2,rank),c(1,3),sd)
  Imps.loco.lin.m = apply(apply(Imps.loco.lin[,,,],1:2,rank),c(1,3),mean)
  Imps.loco.lin.s = apply(apply(Imps.loco.lin[,,,],1:2,rank),c(1,3),sd)
  rel.gh.lin.m = apply(apply(rel.gh.lin[,,,],1:2,rank),c(1,3),mean)
  rel.gh.lin.e.m = apply(apply(rel.gh.lin.e[,,,],1:2,rank),c(1,3),mean)
  rel.gh.lin.s = apply(apply(rel.gh.lin[,,,],1:2,rank),c(1,3),sd)
  rel.gh.lin.e.s = apply(apply(rel.gh.lin.e[,,,],1:2,rank),c(1,3),sd)
  Imps.knoc.lin.m = apply(apply(Imps.knoc.lin[,,,],1:2,rank),c(1,3),mean)
  Imps.knoc.lin.s = apply(apply(Imps.knoc.lin[,,,],1:2,rank),c(1,3),sd)
}
if (fit.rf){
  Imps.perm.rf.m = apply(apply(Imps.perm.rf[,,,],1:2,rank),c(1,3),mean)
  Imps.perm.rf.s = apply(apply(Imps.perm.rf[,,,],1:2,rank),c(1,3),sd)
  Imps.cond.rf.m = apply(apply(Imps.cond.rf[,,,],1:2,rank),c(1,3),mean)
  Imps.cond.rf.s = apply(apply(Imps.cond.rf[,,,],1:2,rank),c(1,3),sd)
  Imps.loco.rf.m = apply(apply(Imps.loco.rf[,,,],1:2,rank),c(1,3),mean)
  Imps.loco.rf.s = apply(apply(Imps.loco.rf[,,,],1:2,rank),c(1,3),sd)
  rel.gh.rf.m =    apply(apply(rel.gh.rf[,,,]   ,1:2,rank),c(1,3),mean)
  rel.gh.rf.e.m =  apply(apply(rel.gh.rf.e[,,,] ,1:2,rank),c(1,3),mean)
  rel.gh.rf.s =    apply(apply(rel.gh.rf[,,,]   ,1:2,rank),c(1,3),sd)
  rel.gh.rf.e.s =  apply(apply(rel.gh.rf.e[,,,] ,1:2,rank),c(1,3),sd)
  Imps.knoc.rf.m = apply(apply(Imps.knoc.rf[,,,],1:2,rank),c(1,3),mean)
  Imps.knoc.rf.s = apply(apply(Imps.knoc.rf[,,,],1:2,rank),c(1,3),sd)
}
if (fit.nn){
  Imps.perm.nn.m = apply(apply(Imps.perm.nn[,,,],1:2,rank),c(1,3),mean)
  Imps.perm.nn.s = apply(apply(Imps.perm.nn[,,,],1:2,rank),c(1,3),sd)
  Imps.cond.nn.m = apply(apply(Imps.cond.nn[,,,],1:2,rank),c(1,3),mean)
  Imps.cond.nn.s = apply(apply(Imps.cond.nn[,,,],1:2,rank),c(1,3),sd)
  Imps.loco.nn.m = apply(apply(Imps.loco.nn[,,,],1:2,rank),c(1,3),mean)
  Imps.loco.nn.s = apply(apply(Imps.loco.nn[,,,],1:2,rank),c(1,3),sd)
  rel.gh.nn.m = apply(apply(rel.gh.nn[,,,],1:2,rank),c(1,3),mean)
  rel.gh.nn.e.m = apply(apply(rel.gh.nn.e[,,,],1:2,rank),c(1,3),mean)
  rel.gh.nn.s = apply(apply(rel.gh.nn[,,,],1:2,rank),c(1,3),sd)
  rel.gh.nn.e.s = apply(apply(rel.gh.nn.e[,,,],1:2,rank),c(1,3),sd)
  Imps.knoc.nn.m = apply(apply(Imps.knoc.nn[,,,],1:2,rank),c(1,3),mean)
  Imps.knoc.nn.s = apply(apply(Imps.knoc.nn[,,,],1:2,rank),c(1,3),sd)
}

printing = FALSE#TRUE
# Now let's look at plots of linear model lm() for different methods
if (fit.lin){
  if (printing) pdf("output_lm_OLS_R.pdf", width = 10, height=5)
  op <- par(mfrow=c(1,2))
  for(k in 1:length(rs)){
    par(mar=c(5,5,4,1))
    plot(Imps.perm.lin.m[,k],type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    lines(Imps.cond.lin.m[,k],type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.lin.m[,k],type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.lin.e.m[,k],type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.lin.m[,k],type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.lin.m[,k],type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
    title(bquote('Linear model fitted by OLS.'~rho==.(rs[k])))
  }
  par(op)
  if (printing) dev.off()
} 

# Now let's look at plots of RF for different methods
if (fit.rf){
  if (printing) pdf("output_rf_R.pdf", width = 10, height=5)
  op <- par(mfrow=c(1,2))
  for(k in 1:length(rs)){
    par(mar=c(5,5,4,1))
    plot(Imps.perm.rf.m[,k],type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    lines(Imps.cond.rf.m[,k],type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.rf.m[,k],type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.rf.e.m[,k],type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.rf.m[,k],type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.rf.m[,k],type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
    title(bquote('Random Forest.'~rho==.(rs[k])))
  }
  par(op)
  if (printing) dev.off()
}

# Now let's look at plots of NNet for different methods
if (fit.nn){
  if (printing) pdf("output_nn_R.pdf", width = 10, height=5)
  op <- par(mfrow=c(1,2))
  for(k in 1:length(rs)){
    par(mar=c(5,5,4,1))
    plot(Imps.perm.nn.m[,k],type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    lines(Imps.cond.nn.m[,k],type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.nn.m[,k],type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.nn.e.m[,k],type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.nn.m[,k],type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.nn.m[,k],type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
    title(bquote('Neural Network.'~rho==.(rs[k])))
  }
  par(op)
  if (printing) dev.off()
}

# Now let's look at plots of RF and NNet with rho=.9 for different methods
if (!X12_linear){
  if (printing) pdf("x12_non_lin_rf_nn_R.pdf", width = 10, height=5)
  op <- par(mfrow=c(1,2))
  #for(k in 1:length(rs)){
  k<- 2
#
  par(mar=c(5,5,4,1))
  plot(Imps.perm.rf.m[,k],type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  lines(Imps.cond.rf.m[,k],type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.rf.m[,k],type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.rf.e.m[,k],type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.rf.m[,k],type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.rf.m[,k],type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
  title('Random Forest, non-linear (x1,x2)')
#
  par(mar=c(5,5,4,1))
  plot(Imps.perm.nn.m[,k],type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  lines(Imps.cond.nn.m[,k],type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.nn.m[,k],type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.nn.e.m[,k],type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.nn.m[,k],type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.nn.m[,k],type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
  title('Neural Network, non-linear (x1,x2)')
  par(op)
  if (printing) dev.off()
}

