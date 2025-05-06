# translation from Python to R from "sim_liang.py"
#

printing <- FALSE # TRUE # 
printing.paper <- TRUE #FALSE # TRUE # 
set.seed(123456) # to reproduce results

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
nsim = 100 # 100 number of repetitions
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
if (fit.lin){
  linmod = vector(mode='list',length=nsim)
  Imps.loco.lin = array(NA,c(nsim,nfeat))
  Imps.perm.lin = array(NA,c(nsim,nfeat))
  #Imps.cond.lin = array(NA,c(nsim,nfeat))
  rel.gh.lin = array(NA,c(nsim,nfeat))
  rel.gh.lin.e = array(NA,c(nsim,nfeat))
  Imps.knoc.lin = array(NA,c(nsim,nfeat))
  MSPEt.lin = array(NA,c(nsim))
}

if (fit.lasso){
  lassomod = vector(mode='list',length=nsim)
  Imps.loco.lasso = array(NA,c(nsim,nfeat))
  Imps.perm.lasso = array(NA,c(nsim,nfeat))
  #Imps.cond.lasso = array(NA,c(nsim,nfeat))
  rel.gh.lasso = array(NA,c(nsim,nfeat))
  rel.gh.lasso.e = array(NA,c(nsim,nfeat))
  Imps.knoc.lasso = array(NA,c(nsim,nfeat))
  MSPEt.lasso = array(NA,c(nsim))
}

if (fit.gam){
  gammod = vector(mode='list',length=nsim)
  Imps.loco.gam = array(NA,c(nsim,nfeat))
  Imps.perm.gam = array(NA,c(nsim,nfeat))
  #Imps.cond.gam = array(NA,c(nsim,nfeat))
  rel.gh.gam = array(NA,c(nsim,nfeat))
  rel.gh.gam.e = array(NA,c(nsim,nfeat))
  Imps.knoc.gam = array(NA,c(nsim,nfeat))
  MSPEt.gam = array(NA,c(nsim))
}

if (fit.rf){#"Imps.orig" (out-of-bag) from randomForest are not recorded
  rfmod = vector(mode='list',length=nsim)
  Imps.loco.rf = array(NA,c(nsim,nfeat))
  Imps.perm.rf = array(NA,c(nsim,nfeat))
  #Imps.cond.rf = array(NA,c(nsim,nfeat))
  Imps.knoc.rf = array(NA,c(nsim,nfeat))
  rel.gh.rf    = array(NA,c(nsim,nfeat))
  rel.gh.rf.e  = array(NA,c(nsim,nfeat))
  MSPEt.rf = array(NA,c(nsim))
}

if (fit.nn){
  nnmod = vector(mode='list',length=nsim)
  Imps.loco.nn = array(NA,c(nsim,nfeat))
  Imps.perm.nn = array(NA,c(nsim,nfeat))
  #Imps.cond.nn = array(NA,c(nsim,nfeat))
  Imps.knoc.nn = array(NA,c(nsim,nfeat))
  rel.gh.nn  = array(NA,c(nsim,nfeat))
  rel.gh.nn.e = array(NA,c(nsim,nfeat))
  MSPEt.nn = array(NA,c(nsim))
}

output.file.name = format(Sys.time(), "%Y_%m_%d_%H_%M_%S")

for(j in 1:nsim){
  print(paste("j=",j," of ",nsim,";  ", Sys.time()))

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
    MSPEt.lin[j] = mean( (yt - yt.hat)^2 )
    linmod[[j]] = t.mod
    time.perm = time.perm + system.time({
      VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.lin[j])
      Imps.perm.lin[j,] = VI.perm
    })
    # time.cond = time.cond + system.time({
    #   VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.lin[j])
    #   Imps.cond.lin[j,] = VI.cond
    # })
    time.knoc = time.knoc + system.time({
      VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.lin[j])
      Imps.knoc.lin[j,] = VI.knoc
    })
    time.gh = time.gh + system.time({
      rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.lin[j],yt.hat)
      rel.gh.lin[j,] <- rel_Gh$relev.ghost
      rel.gh.lin.e[j,] <- rel_Gh$relev.ghost.e
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
    MSPEt.lasso[j] = mean( (yt - yt.hat)^2 )
    lassomod[[j]] = t.mod
    time.perm = time.perm + system.time({
      VI.perm = Vimp.perturb(t.mod,Xt,yt,pertXt=pXt,MSPEt.lasso[j])
      Imps.perm.lasso[j,] = VI.perm
    })
    # time.cond = time.cond + system.time({
    #   VI.cond = Vimp.perturb(t.mod,Xt,yt,pertXt=cXt,MSPEt.lasso[j])
    #   Imps.cond.lasso[j,] = VI.lasso
    # })
    time.knoc = time.knoc + system.time({
      VI.knoc = Vimp.perturb(t.mod,Xt,yt,pertXt=kXt,MSPEt.lasso[j])
      Imps.knoc.lasso[j,] = VI.knoc
    })
    time.gh = time.gh + system.time({
      rel_Gh <- Vimp.perturb.gh(t.mod,Xt,yt,pertXt=gXt,MSPEt.lasso[j],yt.hat)
      rel.gh.lasso[j,] <- rel_Gh$relev.ghost
      rel.gh.lasso.e[j,] <- rel_Gh$relev.ghost.e
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
    MSPEt.gam[j] = mean( (yt - yt.hat)^2 )
    linmod[[j]] = t.mod
    time.perm = time.perm + system.time({
      VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.gam[j])
      Imps.perm.gam[j,] = VI.perm
    })
    # time.cond = time.cond + system.time({
    #   VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.gam[j])
    #   Imps.cond.gam[j,] = VI.cond
    # })
    time.knoc = time.knoc + system.time({
      VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.gam[j])
      Imps.knoc.gam[j,] = VI.knoc
    })
    time.gh = time.gh + system.time({
      rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.gam[j],yt.hat)
      rel.gh.gam[j,] <- rel_Gh$relev.ghost
      rel.gh.gam.e[j,] <- rel_Gh$relev.ghost.e
    })
    time_gam = time_gam + proc.time() - ptm
  }
      
  # Random Forest
  if (fit.rf){
    ptm <- proc.time()
    t.mod = randomForest(X,y)
    rfmod[[j]] = t.mod
    #MSPE = mean( (y - predict(t.mod,X))^2 )
    yt.hat <- predict(t.mod,Xt)
    MSPEt.rf[j] = mean( (yt - yt.hat)^2 )
    time.perm = time.perm + system.time({
      VI.perm = Vimp.perturb(t.mod,Xt,yt,pertXt=pXt,MSPEt.rf[j])
      Imps.perm.rf[j,] = VI.perm
    })
    # time.cond = time.cond + system.time({
    #   VI.cond = Vimp.perturb(t.mod,Xt,yt,pertXt=cXt,MSPEt.rf[j])
    #   Imps.cond.rf[j,] = VI.cond
    # })
    time.knoc = time.knoc + system.time({
      VI.knoc = Vimp.perturb(t.mod,Xt,yt,pertXt=kXt,MSPEt.rf[j])
      Imps.knoc.rf[j,] = VI.knoc
    })
    time.gh = time.gh + system.time({
      rel_Gh <- Vimp.perturb.gh(t.mod,Xt,yt,pertXt=gXt,MSPEt.rf[j],yt.hat)
      rel.gh.rf[j,]   <- rel_Gh$relev.ghost
      rel.gh.rf.e[j,] <- rel_Gh$relev.ghost.e
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
    MSPEt.nn[j] = mean( (yt - yt.hat)^2 )
    linmod[[j]] = t.mod
    time.perm = time.perm + system.time({
      VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.nn[j])
      Imps.perm.nn[j,] = VI.perm
    })
    # time.cond = time.cond + system.time({
    #   VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.nn[j])
    #   Imps.cond.nn[j,] = VI.cond
    # })
    time.knoc = time.knoc + system.time({
      VI.knoc = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=kXt,MSPEt.nn[j])
      Imps.knoc.nn[j,] = VI.knoc
    })
    time.gh = time.gh + system.time({
      rel_Gh <- Vimp.perturb.gh(t.mod,data.frame(Xt),yt,pertXt=gXt,MSPEt.nn[j],yt.hat)
      rel.gh.nn[j,] <- rel_Gh$relev.ghost
      rel.gh.nn.e[j,] <- rel_Gh$relev.ghost.e
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
        Imps.loco.lin[j,l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.lin[j] -1 # relative VI	
        time_lin = time_lin + proc.time() - ptm
      }
      if (fit.lasso){
        ptm <- proc.time()
        cv.lasso.tr <- cv.glmnet(X[,-l], y, nfolds = 10)
        t.mod <- glmnet(X[,-l], y, lambda=cv.lasso.tr$lambda.min)
        yt.hat <- predict(t.mod, newx = Xt[,-l])
        Imps.loco.lasso[j,l] = mean( (yt - yt.hat)^2 )/MSPEt.lasso[j] -1 # relative VI	
        time_lasso = time_lasso + proc.time() - ptm
      }
      if (fit.gam){
        ptm <- proc.time()
        t.mod = gam(fs,simdat)
        Imps.loco.gam[j,l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.gam[j] -1 # relative VI	
        time_gam = time_gam + proc.time() - ptm
      }
      if (fit.rf){
        ptm <- proc.time()
        t.mod = randomForest(X[,-l],y)
        Imps.loco.rf[j,l] = mean(  (yt - predict(t.mod,Xt[,-l]))^2 )/MSPEt.rf[j] -1 # relative VI
        time_rf = time_rf + proc.time() - ptm
      }
      if (fit.nn){
        ptm <- proc.time()
        t.mod = nnetselect(f,data=simdat,ytrue=ytrue,size=2*(nfeat),its=10,linout=TRUE,trace=FALSE)
        #		t.mod = nnet(f,simdat,size=nfeat,maxit=1000,linout=TRUE,trace=FALSE)
        Imps.loco.nn[j,l] = mean( (yt - predict(t.mod,data.frame(Xt[,-l])))^2 )/MSPEt.nn[j] -1 # relative VI
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
  #if ((100*(j/nsim))%%100==0) save.image(paste0('output_simulation/',output.file.name,'.Rdata'),safe=TRUE)
}

save.image(paste0('output_simulation/',output.file.name,'.Rdata'),safe=TRUE)

print(times.VI.methods)
print(times_models)

# Now lets look at some of this: means and standard deviations of importances
if (fit.lin){
  Imps.perm.lin.m = apply(apply(Imps.perm.lin[,],1,rank),1,mean)
  Imps.perm.lin.s = apply(apply(Imps.perm.lin[,],1,rank),1,sd)
  # Imps.cond.lin.m = apply(apply(Imps.cond.lin[,],1,rank),1,mean)
  # Imps.cond.lin.s = apply(apply(Imps.cond.lin[,],1,rank),1,sd)
  Imps.loco.lin.m = apply(apply(Imps.loco.lin[,],1,rank),1,mean)
  Imps.loco.lin.s = apply(apply(Imps.loco.lin[,],1,rank),1,sd)
  rel.gh.lin.m = apply(apply(rel.gh.lin[,],1,rank),1,mean)
  rel.gh.lin.e.m = apply(apply(rel.gh.lin.e[,],1,rank),1,mean)
  rel.gh.lin.s = apply(apply(rel.gh.lin[,],1,rank),1,sd)
  rel.gh.lin.e.s = apply(apply(rel.gh.lin.e[,],1,rank),1,sd)
  Imps.knoc.lin.m = apply(apply(Imps.knoc.lin[,],1,rank),1,mean)
  Imps.knoc.lin.s = apply(apply(Imps.knoc.lin[,],1,rank),1,sd)
}
if (fit.lasso){
  Imps.perm.lasso.m = apply(apply(Imps.perm.lasso[,],1,rank),1,mean)
  Imps.perm.lasso.s = apply(apply(Imps.perm.lasso[,],1,rank),1,sd)
  # Imps.cond.lasso.m = apply(apply(Imps.cond.lasso[,],1,rank),1,mean)
  # Imps.cond.lasso.s = apply(apply(Imps.cond.lasso[,],1,rank),1,sd)
  Imps.loco.lasso.m = apply(apply(Imps.loco.lasso[,],1,rank),1,mean)
  Imps.loco.lasso.s = apply(apply(Imps.loco.lasso[,],1,rank),1,sd)
  rel.gh.lasso.m = apply(apply(rel.gh.lasso[,],1,rank),1,mean)
  rel.gh.lasso.e.m = apply(apply(rel.gh.lasso.e[,],1,rank),1,mean)
  rel.gh.lasso.s = apply(apply(rel.gh.lasso[,],1,rank),1,sd)
  rel.gh.lasso.e.s = apply(apply(rel.gh.lasso.e[,],1,rank),1,sd)
  Imps.knoc.lasso.m = apply(apply(Imps.knoc.lasso[,],1,rank),1,mean)
  Imps.knoc.lasso.s = apply(apply(Imps.knoc.lasso[,],1,rank),1,sd)
}
if (fit.gam){
  Imps.perm.gam.m = apply(apply(Imps.perm.gam[,],1,rank),1,mean)
  Imps.perm.gam.s = apply(apply(Imps.perm.gam[,],1,rank),1,sd)
  # Imps.cond.gam.m = apply(apply(Imps.cond.gam[,],1,rank),1,mean)
  # Imps.cond.gam.s = apply(apply(Imps.cond.gam[,],1,rank),1,sd)
  Imps.loco.gam.m = apply(apply(Imps.loco.gam[,],1,rank),1,mean)
  Imps.loco.gam.s = apply(apply(Imps.loco.gam[,],1,rank),1,sd)
  rel.gh.gam.m = apply(apply(rel.gh.gam[,],1,rank),1,mean)
  rel.gh.gam.e.m = apply(apply(rel.gh.gam.e[,],1,rank),1,mean)
  rel.gh.gam.s = apply(apply(rel.gh.gam[,],1,rank),1,sd)
  rel.gh.gam.e.s = apply(apply(rel.gh.gam.e[,],1,rank),1,sd)
  Imps.knoc.gam.m = apply(apply(Imps.knoc.gam[,],1,rank),1,mean)
  Imps.knoc.gam.s = apply(apply(Imps.knoc.gam[,],1,rank),1,sd)
}
if (fit.rf){
  Imps.perm.rf.m = apply(apply(Imps.perm.rf[,],1,rank),1,mean)
  Imps.perm.rf.s = apply(apply(Imps.perm.rf[,],1,rank),1,sd)
  # Imps.cond.rf.m = apply(apply(Imps.cond.rf[,],1,rank),1,mean)
  # Imps.cond.rf.s = apply(apply(Imps.cond.rf[,],1,rank),1,sd)
  Imps.loco.rf.m = apply(apply(Imps.loco.rf[,],1,rank),1,mean)
  Imps.loco.rf.s = apply(apply(Imps.loco.rf[,],1,rank),1,sd)
  rel.gh.rf.m =    apply(apply(rel.gh.rf[,]   ,1,rank),1,mean)
  rel.gh.rf.e.m =  apply(apply(rel.gh.rf.e[,] ,1,rank),1,mean)
  rel.gh.rf.s =    apply(apply(rel.gh.rf[,]   ,1,rank),1,sd)
  rel.gh.rf.e.s =  apply(apply(rel.gh.rf.e[,] ,1,rank),1,sd)
  Imps.knoc.rf.m = apply(apply(Imps.knoc.rf[,],1,rank),1,mean)
  Imps.knoc.rf.s = apply(apply(Imps.knoc.rf[,],1,rank),1,sd)
}
if (fit.nn){
  Imps.perm.nn.m = apply(apply(Imps.perm.nn[,],1,rank),1,mean)
  Imps.perm.nn.s = apply(apply(Imps.perm.nn[,],1,rank),1,sd)
  # Imps.cond.nn.m = apply(apply(Imps.cond.nn[,],1,rank),1,mean)
  # Imps.cond.nn.s = apply(apply(Imps.cond.nn[,],1,rank),1,sd)
  Imps.loco.nn.m = apply(apply(Imps.loco.nn[,],1,rank),1,mean)
  Imps.loco.nn.s = apply(apply(Imps.loco.nn[,],1,rank),1,sd)
  rel.gh.nn.m = apply(apply(rel.gh.nn[,],1,rank),1,mean)
  rel.gh.nn.e.m = apply(apply(rel.gh.nn.e[,],1,rank),1,mean)
  rel.gh.nn.s = apply(apply(rel.gh.nn[,],1,rank),1,sd)
  rel.gh.nn.e.s = apply(apply(rel.gh.nn.e[,],1,rank),1,sd)
  Imps.knoc.nn.m = apply(apply(Imps.knoc.nn[,],1,rank),1,mean)
  Imps.knoc.nn.s = apply(apply(Imps.knoc.nn[,],1,rank),1,sd)
}

# Now let's look at plots of linear model lm() for different methods
if (fit.lin){
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lin.m,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lin.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lin.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lin.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lin.e.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lin.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c(expression(pi),'L','G','Ge','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
  title(bquote('Linear Model Importance'))
} 
# Now let's look at plots of linear model lm() for different methods
if (fit.lin){
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lin.s,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lin.s,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lin.s,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lin.s,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lin.e.s,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lin.s,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c(expression(pi),'L','G','Ge','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
  title(bquote('Linear Model Importance (sd)'))
} 

# Now let's look at plots of linear model fitted by lasso for different methods
if (fit.lasso){
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lasso.m,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lasso.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lasso.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lasso.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lasso.e.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lasso.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c(expression(pi),'L','G','Ge','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
  title(bquote('Lasso Linear Model Importance'))
} 

# Now let's look at plots of linear model fitted by lasso for different methods
if (fit.lasso){
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lasso.s,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lasso.s,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lasso.s,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lasso.s,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lasso.e.s,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lasso.s,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c(expression(pi),'L','G','Ge','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
  title(bquote('Lasso Linear Model Importance (sd)'))
} 

printing <- FALSE # TRUE # FALSE #
if (1==1){
  if (printing) pdf("sim_lin_mod_100_var.pdf", width = 10, height=5)
  op <- par(mfrow=c(1,2))
  
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lin.m,type='b',pch=1,cex=1.5,lwd=2,
       xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lin.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lin.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lin.e.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lin.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lin.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c('P','L','G','Gp','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1,lwd=2,bg="white")
  title(bquote('Linear model estimated by OLS'))
  
  par(mar=c(5,5,4,1))
  plot(Imps.perm.lasso.m,type='b',pch=1,cex=1.5,lwd=2,
       xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lasso.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lasso.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lasso.e.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lasso.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lasso.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c('P','L','G','Gp','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1,lwd=2,bg="white")
  title(bquote('Linear model estimated by lasso'))
  
  par(op)
  if (printing) dev.off()
}

if (printing) pdf("sim_lm_100v_gh_rel.pdf", width = 10, height=5)
boxplot.matrix(rel.gh.lin.e,main="Linear model, OLS. Relev by ghost variables",
               xlab="Features",ylab="Relev by ghost vars",
               cex.main=1.5,cex.lab=1.5)
if (printing) dev.off()

# Now let's look at plots of model gam() for different methods
if (fit.gam){
    par(mar=c(5,5,4,1))
    plot(Imps.perm.gam.m,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    #lines(Imps.cond.gam.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.gam.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.gam.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.gam.e.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.gam.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c(expression(pi),'L','G','Ge','Nk'),
           pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
    title(bquote('GAM Importance'))
} 

# Now let's look at plots of RF for different methods
if (fit.rf){
    par(mar=c(5,5,4,1))
    plot(Imps.perm.rf.m,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    #lines(Imps.cond.rf.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.rf.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.rf.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.rf.e.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.rf.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c(expression(pi),'L','G','Ge','Nk'),
           pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
    title(bquote('Random Forest Importance'))
}

# Now let's look at plots of NNet for different methods
if (fit.nn){
    par(mar=c(5,5,4,1))
    plot(Imps.perm.nn.m,type='b',pch=1,cex=1.5,lwd=2,xlab='feature',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    #lines(Imps.cond.nn.m,type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.nn.m,type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.nn.m,type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.nn.e.m,type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.nn.m,type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c(expression(pi),'L','G','Ge','Nk'),
           pch=c(1,3:6),col=c(1,3:6),cex=1.5,lwd=2)
    title(bquote('Neural Network Importance'))
}

boxplot(MSPEt.lin/var(y),MSPEt.lasso/var(y))

op <- par(mfrow=c(2,2))
boxplot.matrix(Imps.perm.lin,main="Linear model OLS. Relev by permut")
boxplot.matrix(Imps.loco.lin,main="Linear model OLS. Relev by loco")
#boxplot.matrix(rel.gh.lin,main="Linear model OLS. Relev by ghost")
boxplot.matrix(rel.gh.lin.e,main="Linear model OLS. Relev by ghost-e")
boxplot.matrix(Imps.knoc.lin,main="Linear model OLS. Relev by knockoffs")
par(op)

op <- par(mfrow=c(2,2))
boxplot.matrix(Imps.perm.lasso,main="Linear model lasso. Relev by permut")
boxplot.matrix(Imps.loco.lasso,main="Linear model lasso. Relev by loco")
#boxplot.matrix(rel.gh.lasso,main="Linear model lasso. Relev by ghost")
boxplot.matrix(rel.gh.lasso.e,main="Linear model lasso. Relev by ghost-e")
boxplot.matrix(Imps.knoc.lasso,main="Linear model lasso. Relev by knockoffs")
par(op)

op <- par(mfrow=c(2,2))
plot(apply(Imps.perm.lin,2,sd),ylab="sd(relev)",main="Linear model OLS. Relev by permut")
plot(apply(Imps.loco.lin,2,sd),ylab="sd(relev)",main="Linear model OLS. Relev by loco",ylim=c(0,.11))
plot(apply(rel.gh.lin.e,2,sd),ylab="sd(relev)",main="Linear model OLS. Relev by ghost-e",ylim=c(0,.11))
plot(apply(Imps.knoc.lin,2,sd),ylab="sd(relev)",main="Linear model OLS. Relev by knockoffs",ylim=c(0,.11))
par(op)

op <- par(mfrow=c(2,2))
plot(apply(Imps.perm.lasso,2,sd),ylab="sd(relev)",main="Linear model lasso. Relev by permut")
plot(apply(Imps.loco.lasso,2,sd),ylab="sd(relev)",main="Linear model lasso. Relev by loco",ylim=c(0,.07))
plot(apply(rel.gh.lasso.e,2,sd),ylab="sd(relev)",main="Linear model lasso. Relev by ghost-e",ylim=c(0,.07))
plot(apply(Imps.knoc.lasso,2,sd),ylab="sd(relev)",main="Linear model lasso. Relev by knockoffs",ylim=c(0,.07))
par(op)

op <- par(mfrow=c(2,2))
boxplot(#apply(Imps.perm.lin,2,sd)/apply(rel.gh.lin.e,2,sd),
  apply(Imps.loco.lin,2,sd)/apply(rel.gh.lin.e,2,sd),
  apply(Imps.knoc.lin,2,sd)/apply(rel.gh.lin.e,2,sd))
plot(apply(Imps.perm.lin,2,sd)/apply(rel.gh.lin.e,2,sd));abline(h=1,lty=2)
plot(apply(Imps.loco.lin,2,sd)/apply(rel.gh.lin.e,2,sd),ylim=c(1,1.6));abline(h=1,lty=2)
plot(apply(Imps.knoc.lin,2,sd)/apply(rel.gh.lin.e,2,sd),ylim=c(1,3));abline(h=1,lty=2)
par(op)

op <- par(mfrow=c(2,2))
boxplot(#apply(Imps.perm.lasso,2,sd)/apply(rel.gh.lin.e,2,sd),
  apply(Imps.loco.lasso,2,sd)/apply(rel.gh.lasso.e,2,sd),
  apply(Imps.knoc.lasso,2,sd)/apply(rel.gh.lasso.e,2,sd))
plot(apply(Imps.perm.lasso,2,sd)/apply(rel.gh.lasso.e,2,sd));abline(h=1,lty=2)
plot(apply(Imps.loco.lasso,2,sd)/apply(rel.gh.lasso.e,2,sd),ylim=c(1,7));abline(h=1,lty=2)
plot(apply(Imps.knoc.lasso,2,sd)/apply(rel.gh.lasso.e,2,sd),ylim=c(1,3));abline(h=1,lty=2)
par(op)


