# Relevance Matrix for one dataset coming from 
# the simulation study
# "sim_Ex3_Variations.R"
#
##################
library(glmnet)
printing <- FALSE # TRUE # 
printing.paper <- TRUE #FALSE # TRUE # 
# seed <- 123456 
seed <- 13579
set.seed(seed) # to reproduce results

gen_Ex3_Variations <- function(n=2000, p1=5, p2=45, p3=50){
  sigma.1 <- 1 # sd for the p1 variables x1.1,...,x1.p1
  sigma.2 <- 1 # sd for the p2 variables x2.1,...,x2.p2
  sigma.3 <- 2 # sd for the p3 variables x3.1,...,x3.p3
  
  sigma.eps <- 1 # residual sd for defining y
  
  # rho.1 = 0
  rho.2 <- .95 # correlation between p2 variables
  rho.3 <- 0#.95 # correlation between p3 variables
  
  beta1 <- .5 # coef. of z1=x1.1+...+x1.p1
  beta2 <-  1 # coef. of z2=x2.1+...+x2.p2
  beta3 <- .1 # coef. of z3=x3.1+...+x3.p3
  
  # Generating the p1 variables
  X1 <- sigma.1 * matrix(rnorm(n*p1),ncol=p1)
  z1 <- apply(X1,1,sum)
  
  # Generating the p2 variables
  Sigma.2 <- matrix(rho.2, nrow=p2, ncol=p2)
  diag(Sigma.2) <- 1
  eig.Sigma.2 <- eigen(Sigma.2)
  sqrt.Sigma.2 <- eig.Sigma.2$vectors %*% diag(eig.Sigma.2$values^.5) %*% t(eig.Sigma.2$vectors)
  X2 <- sigma.2 * matrix(rnorm(n*p2),ncol=p2) %*% sqrt.Sigma.2
  z2 <- apply(X2,1,sum)
  #z2 <- as.numeric(X2 %*% (1:p2))/p2
  
  # Generating the p3 variables
  Sigma.3 <- matrix(rho.3, nrow=p3, ncol=p3)
  diag(Sigma.3) <- 1
  eig.Sigma.3 <- eigen(Sigma.3)
  sqrt.Sigma.3 <- eig.Sigma.3$vectors %*% diag(eig.Sigma.3$values^.5) %*% t(eig.Sigma.3$vectors)
  X3 <- sigma.3 * matrix(rnorm(n*p3),ncol=p3) %*% sqrt.Sigma.3
  z3 <- apply(X3,1,sum)
  
  
  # defining the response variable
  y <- beta1*z1 + beta2*z2 + beta3*z3 + rnorm(n,sd=sigma.eps)
  
  X <- cbind(X1,X2,X3)
  #colnames(X) <- c( paste0("x1.",1:p1), paste0("x2.",1:p2) , paste0("x3.",1:p3) )
  #yX <- as.data.frame(cbind(y,X))
  return(list(X=X,y=y))
}


source("../Hooker_et_al_2021/aux_functs_simul_Ex_Hooker_et_al.R")

# 
# model fitting for creating the ghost variables
ghost_model_linear <- TRUE # TRUE # When TRUEE, lm() is used for ghost variables 
ghost_model_gam <- !ghost_model_linear # When TRUE, gam() is used for ghost variables 

# simulation parameters:
nsim = 1 # 100 number of repetitions
n <- 1000  # training set size
nt = n/2  # test set size
p1=5; p2=45; p3=50 # number of each type of explanatory variables
nfeat <- p1+p2+p3 # number of explanatory variables

# models to be fitted
fit.lin <- TRUE # FALSE #
fit.lasso<-TRUE # FALSE #
fit.gam <- FALSE # TRUE # 
fit.rf  <- FALSE #TRUE # FALSE #
fit.nn  <- FALSE # TRUE #   

model.names <-c("lin","lasso","gam","rf","nn")[c(fit.lin,fit.lasso,fit.gam,fit.rf,fit.nn)]
time_lin=0
time_lasso=0
time_gam=0
time_rf=0
time_nn=0

# Relevance/Importance methods to be computed
do_loco <- TRUE #  FALSE # 
time.perm <- 0
#time.cond <- 0
time.loco <- 0
time.gh <- 0
time.knoc <- 0

##############

# training set
Xy <- gen_Ex3_Variations(n=n, p1=p1, p2=p2, p3=p3)
X <- Xy$X
y <- Xy$y


# test set
Xyt <- gen_Ex3_Variations(n=nt,  p1=p1, p2=p2, p3=p3)
Xt <- Xyt$X
yt <- Xyt$y

# Constructing the "perturbed" test sets by
# random permutations, #theoretical conditional distribution,
# knockoffs, or ghost variables:
time.perm=time.perm + system.time({
  pXt = random.perm(Xt)    # random permutations
})
# time.cond=time.cond + system.time({
#   cXt = cond.distrib(Xt,r) # theoretical conditional distribution
# })
time.knoc=time.knoc + system.time({
  kXt = knockoff_MX(Xt)    # knockoffs
})
time.gh=time.gh + system.time({
  if (ghost_model_linear){
    gXt = ghost_var_lm(Xt)      # ghost variables fitted by lm()
  }else{
    gXt = ghost_var_gam(Xt)     # ghost variables fitted by gam()
  }
})

simdat = data.frame(y,X)
f = formula(paste('y~',paste(names(simdat)[-1],collapse='+')))

# Linear Model, fitted by OLS
if (fit.lin){
  ptm <- proc.time()
  t.mod = lm(f,data=simdat)
  #MSPE = mean( (y - predict(t.mod,data.frame(X)))^2 )
  yt.hat <- predict(t.mod,data.frame(Xt))
  MSPEt.lin = mean( (yt - yt.hat)^2 )
  linmod = t.mod
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.lin)
    Imps.perm.lin = VI.perm
  })
  # time.cond = time.cond + system.time({
  #   VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.lin)
  #   Imps.cond.lin = VI.cond
  # })
  time.knoc = time.knoc + system.time({
    VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.lin)
    Imps.knoc.lin = VI.knoc
  })
  time.gh = time.gh + system.time({
    rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.lin,yt.hat)
    rel.gh.lin <- rel_Gh$relev.ghost
    rel.gh.lin.e <- rel_Gh$relev.ghost.e
  })
  time_lin = time_lin + proc.time() - ptm
}

# Linear Model, fitted by LASSO
if (fit.lasso){
  ptm <- proc.time()
  cv.lasso.tr <- cv.glmnet(X, y, nfolds = 10)
  t.mod <- glmnet(X, y, lambda=cv.lasso.tr$lambda.min)
  #MSPE = mean( (y - predict(t.mod,data.frame(X)))^2 )
  yt.hat <- predict(t.mod, newx = Xt)
  MSPEt.lasso = mean( (yt - yt.hat)^2 )
  lassomod = t.mod
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,Xt,yt,pertXt=pXt,MSPEt.lasso)
    Imps.perm.lasso = VI.perm
  })
  # time.cond = time.cond + system.time({
  #   VI.cond = Vimp.perturb(t.mod,Xt,yt,pertXt=cXt,MSPEt.lasso)
  #   Imps.cond.lasso = VI.lasso
  # })
  time.knoc = time.knoc + system.time({
    VI.knoc = Vimp.perturb(t.mod,Xt,yt,pertXt=kXt,MSPEt.lasso)
    Imps.knoc.lasso = VI.knoc
  })
  time.gh = time.gh + system.time({
    rel_Gh <- Vimp.perturb.gh(t.mod,Xt,yt,pertXt=gXt,MSPEt.lasso,yt.hat)
    rel.gh.lasso <- rel_Gh$relev.ghost
    rel.gh.lasso.e <- rel_Gh$relev.ghost.e
  })
  time_lasso = time_lasso + proc.time() - ptm
}

# GAM
if (fit.gam){
  ptm <- proc.time()
  vars <- names(simdat)
  s.terms <- paste0("s(",vars,")")
  fs <- as.formula(paste0(vars[1],"~",paste(s.terms[-1],collapse="+")))        
  t.mod = gam(fs,data=simdat)
  #MSPE = mean( (y - predict(t.mod,data.frame(X)))^2 )
  yt.hat <- predict(t.mod,data.frame(Xt))
  MSPEt.gam = mean( (yt - yt.hat)^2 )
  linmod = t.mod
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.gam)
    Imps.perm.gam = VI.perm
  })
  # time.cond = time.cond + system.time({
  #   VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.gam)
  #   Imps.cond.gam = VI.cond
  # })
  time.knoc = time.knoc + system.time({
    VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.gam)
    Imps.knoc.gam = VI.knoc
  })
  time.gh = time.gh + system.time({
    rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.gam,yt.hat)
    rel.gh.gam <- rel_Gh$relev.ghost
    rel.gh.gam.e <- rel_Gh$relev.ghost.e
  })
  time_gam = time_gam + proc.time() - ptm
}

# Random Forest
if (fit.rf){
  ptm <- proc.time()
  t.mod = randomForest(X,y)
  rfmod = t.mod
  #MSPE = mean( (y - predict(t.mod,X))^2 )
  yt.hat <- predict(t.mod,Xt)
  MSPEt.rf = mean( (yt - yt.hat)^2 )
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,Xt,yt,pertXt=pXt,MSPEt.rf)
    Imps.perm.rf = VI.perm
  })
  # time.cond = time.cond + system.time({
  #   VI.cond = Vimp.perturb(t.mod,Xt,yt,pertXt=cXt,MSPEt.rf)
  #   Imps.cond.rf = VI.cond
  # })
  time.knoc = time.knoc + system.time({
    VI.knoc = Vimp.perturb(t.mod,Xt,yt,pertXt=kXt,MSPEt.rf)
    Imps.knoc.rf = VI.knoc
  })
  time.gh = time.gh + system.time({
    rel_Gh <- Vimp.perturb.gh(t.mod,Xt,yt,pertXt=gXt,MSPEt.rf,yt.hat)
    rel.gh.rf   <- rel_Gh$relev.ghost
    rel.gh.rf.e <- rel_Gh$relev.ghost.e
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
  MSPEt.nn = mean( (yt - yt.hat)^2 )
  linmod = t.mod
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.nn)
    Imps.perm.nn = VI.perm
  })
  # time.cond = time.cond + system.time({
  #   VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.nn)
  #   Imps.cond.nn = VI.cond
  # })
  time.knoc = time.knoc + system.time({
    VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.nn)
    Imps.knoc.nn = VI.knoc
  })
  time.gh = time.gh + system.time({
    rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.nn,yt.hat)
    rel.gh.nn <- rel_Gh$relev.ghost
    rel.gh.nn.e <- rel_Gh$relev.ghost.e
  })
  time_nn = time_nn + proc.time() - ptm
}

# Now versions of re-training models: only loco has been implemented
if (do_loco){
  ptm.loco <- proc.time()
  Imps.loco.lin <- numeric(nfeat)
  Imps.loco.lasso <- numeric(nfeat)
  Imps.loco.gam <- numeric(nfeat)
  Imps.loco.rf <- numeric(nfeat)
  Imps.loco.nn <- numeric(nfeat)
  for(l in 1:ncol(X)){
    # Dropping covariates: loco, leave-one-covariate-out
    simdat = data.frame(y,X[,-l])
    f = formula(paste('y~',paste(names(simdat)[-1],collapse='+')))
    if (fit.lin){
      ptm <- proc.time()
      t.mod = lm(f,simdat)
      Imps.loco.lin[l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.lin -1 # relative VI	
      time_lin = time_lin + proc.time() - ptm
    }
    if (fit.lasso){
      ptm <- proc.time()
      cv.lasso.tr <- cv.glmnet(X[,-l], y, nfolds = 10)
      t.mod <- glmnet(X[,-l], y, lambda=cv.lasso.tr$lambda.min)
      yt.hat <- predict(t.mod, newx = Xt[,-l])
      Imps.loco.lasso[l] = mean( (yt - yt.hat)^2 )/MSPEt.lasso -1 # relative VI	
      time_lasso = time_lasso + proc.time() - ptm
    }
    if (fit.gam){
      ptm <- proc.time()
      t.mod = gam(fs,simdat)
      Imps.loco.gam[l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.gam -1 # relative VI	
      time_gam = time_gam + proc.time() - ptm
    }
    if (fit.rf){
      ptm <- proc.time()
      t.mod = randomForest(X[,-l],y)
      Imps.loco.rf[l] = mean(  (yt - predict(t.mod,Xt[,-l]))^2 )/MSPEt.rf -1 # relative VI
      time_rf = time_rf + proc.time() - ptm
    }
    if (fit.nn){
      ptm <- proc.time()
      t.mod = nnetselect(f,data=simdat,ytrue=ytrue,size=2*(nfeat),its=10,linout=TRUE,trace=FALSE)
      #		t.mod = nnet(f,simdat,size=nfeat,maxit=1000,linout=TRUE,trace=FALSE)
      Imps.loco.nn[l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.nn -1 # relative VI
      time_nn = time_nn + proc.time() - ptm
    }
  }
  time.loco <- time.loco + proc.time() - ptm.loco
}
times.VI.methods <- data.frame(
  time.perm=time.perm[3],
  #time.cond=time.cond[3],
  time.loco=time.loco[3],
  time.gh=time.gh[3],
  time.knoc=time.knoc[3]
)
times_models <- data.frame(
  time_lin=time_lin[3],
  time_lasso=time_lasso[3],
  time_gam=time_gam[3],
  time_rf=time_rf[3],
  time_nn=time_nn[3]
)

print(times.VI.methods)
print(times_models)

# Now lets look at some of this: means and standard deviations of importances
if (fit.lin){
  Imps.perm.lin.r = rank(Imps.perm.lin)
  Imps.loco.lin.r = rank(Imps.loco.lin)
  rel.gh.lin.r = rank(rel.gh.lin)
  rel.gh.lin.e.r = rank(rel.gh.lin.e)
  Imps.knoc.lin.r = rank(Imps.knoc.lin)
}
if (fit.lasso){
  Imps.perm.lasso.r = rank(Imps.perm.lasso)
  Imps.loco.lasso.r = rank(Imps.loco.lasso)
  rel.gh.lasso.r =    rank(rel.gh.lasso,1:2)
  rel.gh.lasso.e.r =  rank(rel.gh.lasso.e,1:2)
  Imps.knoc.lasso.r = rank(Imps.knoc.lasso)
}
if (fit.gam){
  Imps.perm.gam.r = rank(Imps.perm.gam)
  Imps.loco.gam.r = rank(Imps.loco.gam)
  rel.gh.gam.r =    rank(rel.gh.gam,1:2)
  rel.gh.gam.e.r =  rank(rel.gh.gam.e,1:2)
  Imps.knoc.gam.r = rank(Imps.knoc.gam)
}
if (fit.rf){
  Imps.perm.rf.r = rank(Imps.perm.rf)
  Imps.loco.rf.r = rank(Imps.loco.rf)
  rel.gh.rf.r =    rank(rel.gh.rf,1:2)
  rel.gh.rf.e.r =  rank(rel.gh.rf.e,1:2)
  Imps.knoc.rf.r = rank(Imps.knoc.rf)
}
if (fit.nn){
  Imps.perm.nn.r = rank(Imps.perm.nn)
  Imps.loco.n.rn = rank(Imps.loco.nn)
  rel.gh.nn.r = rank(rel.gh.nn)
  rel.gh.nn.e.r = rank(rel.gh.nn.e)
  Imps.knoc.nn.r = rank(Imps.knoc.nn)
}

printing = FALSE#TRUE
op <- par(mfrow=c(2,1))
# Now let's look at plots of linear model lm() for different methods
if (fit.lin){
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lin.r,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lin.r,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lin.r,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lin.r,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lin.e.r,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lin.r,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c(expression(pi),'L','G','Ge','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=.5,lwd=2)
  title(bquote('Linear Model Importance'))
} 

# Now let's look at plots of linear model fitted by lasso for different methods
if (fit.lasso){
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lasso.r,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lasso.r,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lasso.r,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lasso.r,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lasso.e.r,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lasso.r,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c(expression(pi),'L','G','Ge','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=.5,lwd=2)
  title(bquote('Lasso Linear Model Importance'))
} 
par(op)

############ Relevance matrix
# variable relevance matrix
library(mgcv)
library(ggplot2)
library(grid)
library(maptools)# For pointLabel
source("../relev.ghost.var.R")
source("../relev_GhVar_glmnet.R")
#source("../relev.rand.perm.R")

model.relev <- linmod
newdata <- data.frame(y=yt,Xt) 
file_pdf <- "Rel_Mat_one_case_Ex_3_lm.pdf"
relev.ghost.lm.out <- relev.ghost.var(model=model.relev, 
                                   newdata = newdata,
                                   func.model.ghost.var= lm)
# if (printing) pdf(file_pdf,height = 8, width = 16)
# plot.Matrix.relev.ghost.var(relev.ghost.lm.out, n1=n, ncols.plot = 5)
# if (printing) dev.off()

model.relev <- lassomod 
newdata <- data.frame(Xt)
file_pdf <- "Rel_Mat_one_case_Ex_3_lasso.pdf"
relev.ghost.lasso.out <- relev_GhVar_glmnet(model.relev, Xt, yt,
                                   func.model.ghost.var= "lm")
# if (printing) pdf(file_pdf,height = 8, width = 16)
# plot.Matrix.relev.ghost.var(relev.ghost.lasso.out, n1=n, ncols.plot = 5)
# if (printing) dev.off()

save.image(file=paste0("output_simulation/one_case_Ex3_seed_",seed,".Rdata"))
