library(randomForest)
library(nnet)
library(knockoff)

# 1-hidden layer Neural Network with nnet::nnet
# Code from Hooker et al. (2021)
nnetselect = function(f,data,ytrue,size,its=10,linout=TRUE,trace=FALSE)
{
	vals = rep(NA,its)
	mods = vector(mode='list',length=its)
	
	tmod = nnet(f,data,size=size,linout=linout,trace=trace)
	mods[[1]] = tmod
	vals[1] = mean(  (tmod$fitted.values - ytrue)^2 ) # Attention: ytrue is used to select the best fit!
	
	for(i in 2:its){
	  mods[[i]] = nnet(f,data,size=size,linout=linout,trace=trace,Wts=mods[[i-1]]$wts)
	  vals[i] = mean( (mods[[i]]$fitted.values - ytrue)^2 )
	}
#	print(vals)
	i = which.min(vals)
#	print(i)
	return(mods[[i]])
}

# random permutations and ("knockoff" or "conditional")
# Code from Hooker et al. (2021) slightly modified (relative VI)
Vimp = function(mod,X,y,j = 1:ncol(X),cX=NULL,MSPE=NULL){
	VI = rep(NA,length(j))
	kVI = rep(NA,length(j))
	#MSPE = mean( (y - predict(mod,X))^2 )
	for(k in 1:length(j)){
		Xp = X
		Xp[,j[k]] = sample(X[,j[k]],replace=FALSE)
		VI[k] = mean( (y - predict(mod,Xp))^2 )/MSPE -1 # relative VI
		if(!is.null(cX)){
			Xp = X
			Xp[,j[k]] = cX[,j[k]]
			kVI[k] = mean( (y - predict(mod,Xp))^2 )/MSPE -1 # relative VI
		}
	}
	return(list(MSPE=MSPE, VI = VI, kVI=kVI))
}

# random permutations only
# Code from Hooker et al. (2021) slightly modified (relative VI)
Vimp.perm = function(mod,X,y,j = 1:ncol(X), MSPE=NULL){
  VI = rep(NA,length(j))
  #MSPE = mean( (y - predict(mod,X))^2 )
  for(k in 1:length(j)){
    Xp = X
    Xp[,j[k]] = sample(X[,j[k]],replace=FALSE)
    VI[k] = mean( (y - predict(mod,Xp))^2 )/MSPE -1 # relative VI
  }
  return(VI)
}

# ("knockoff" or "conditional") only
# Code from Hooker et al. (2021) slightly modified (relative VI)
Vimp.cond = function(mod,X,y,j = 1:ncol(X),cX,MSPE=NULL){
  kVI = rep(NA,length(j))
  #if (is.null(MSPE)){MSPE = mean( (y - predict(mod,X))^2 )}
  for(k in 1:length(j)){
    Xp = X
    Xp[,j[k]] = cX[,j[k]]
    kVI[k] = mean( (y - predict(mod,Xp))^2 )/MSPE -1 # relative VI
  }
  return(kVI)
}

## This is the "knockoff" function defined in "11222_2021_10057_MOESM5_ESM.r".
## It does not provide valid knockoffs.
## Instead it provides values of the conditional distribution of
## each explanatory variable given the other
## It should be better to call this function "cond.distrib".
## This function is valid only for this example.
# Code from Hooker et al. (2021) slightly modified (function name)
cond.distrib.1 = function(X,r){
  Z = qnorm(X[,1:2])
  kz1 = rnorm(nrow(X),mean=r*Z[,2], sd = sqrt(1-r^2))
  kz2 = rnorm(nrow(X),mean=r*Z[,1], sd = sqrt(1-r^2))
  return( cbind( pnorm(kz1),pnorm(kz2), matrix(runif(nrow(X)*(ncol(X)-2)),nrow(X),ncol(X)-2) ) )
}

## Now, Model-X Gaussian knockoffs from library knockoff are used instead: 
knockoff_MX = function(X){
   require(knockoff)
   return(create.gaussian(X,colMeans(X),cov(X)))
}

### NEW CODE
# 
# Fits all lm() column models to compute the Relevance by Ghost Variables
ghost_var_lm =  function(X){
  n <- dim(X)[1]
  p <- dim(X)[2]
  X <- cbind(as.matrix(X),rep(1,n))#cte column added at the end of X
  X.gh <- matrix(NA,n,p)
  for (j in (1:p)){
    #X.gh[,j] <- lm(X[,j]~X[,-j]-1)$fitted.values
    X_no_j <- X[,-j]
    X.gh[,j]<-X_no_j%*%solve(t(X_no_j)%*%X_no_j)%*%t(X_no_j)%*%X[,j]
  }
  return(X.gh)  
}

# Fits all gam() column models to compute the Relevance by Ghost Variables
ghost_var_gam =  function(X){
  require(mgcv)
  n <- dim(X)[1]
  p <- dim(X)[2]
  Xdf <- as.data.frame(X)
  vars <- names(Xdf)
  s.terms <- paste0("s(",vars,")")
  X.gh <- matrix(NA,n,p)
  for (j in (1:p)){
    form_j <- as.formula(paste0(vars[j],"~",paste(s.terms[-j],collapse="+")))
    X.gh[,j] <- gam(form_j, data=Xdf)$fitted.values
  }
  return(X.gh)  
}

# random permutations only
random.perm = function(X){
  n <- dim(X)[1]
  p <- dim(X)[2]
  X.perm <- matrix(NA,n,p)
  for(j in 1:p){
    X.perm[,j] = sample(X[,j],replace=FALSE)
  }
  return(X.perm)
}

Vimp.perturb = function(model,Xt,yt,pertXt,MSPEt){
  # pertXt: Matrix with the "perturbed" columns of Xt
  p <- dim(Xt)[2]
  VI = numeric(p)
  #if (is.null(MSPE)){MSPE = mean( (y - predict(mod,X))^2 )}
  for(j in 1:p){
    Xp = Xt
    Xp[,j] = pertXt[,j]
    yt.hat.j <- predict(model,Xp)
    VI[j] = mean( (yt - yt.hat.j)^2 )/MSPEt -1 # relative VI
  }
  return(VI)
}

Vimp.perturb.pred.err = function(model,Xt,yt,pertXt,MSPEt,yt.hat){
  # pertXt: Matrix with the "perturbed" columns of Xt
  p <- dim(Xt)[2]
  VI.pred = numeric(p)
  VI.err = numeric(p)
  #if (is.null(MSPE)){MSPE = mean( (y - predict(mod,X))^2 )}
  for(j in 1:p){
    Xp = Xt
    Xp[,j] = pertXt[,j]
    yt.hat.j <- predict(model,Xp)
    VI.pred[j] = mean((yt.hat - yt.hat.j)^2)/MSPEt    # relative VI based on predictions
    VI.err[j]  = mean((yt     - yt.hat.j)^2)/MSPEt -1 # relative VI based on errors
  }
  return(list(VI.pred=VI.pred, VI.err=VI.err))
}

Vimp.perturb.gh = function(model,Xt,yt,pertXt,MSPEt,yt.hat){
  # pertXt: Matrix with the "perturbed" columns of Xt
  p <- dim(Xt)[2]
  relev.ghost <- relev.ghost.e <- numeric(p)
  #if (is.null(MSPE)){MSPE = mean( (y - predict(mod,X))^2 )}
  for(j in 1:p){
    Xp = Xt
    Xp[,j] = pertXt[,j]
    yt.hat.j <- predict(model,Xp)
    relev.ghost[j]   <- mean((yt.hat- yt.hat.j)^2)/MSPEt
    relev.ghost.e[j] <- mean((yt    - yt.hat.j)^2)/MSPEt - 1
  }
  return(list(relev.ghost=relev.ghost,relev.ghost.e=relev.ghost.e))
}

gen_2_vars_not_normal<-function(n=100,r=.9){
  theta <- runif(n,0,2*pi)
  R <- runif(n,r,1)
  X<- R*cbind(sin(theta),cos(theta))
  X[,2] <- sign(X[,1])*abs(X[,2])
  X <- .5 + .5*X
  return(X)
}

cond.distrib = function(X,r=.9,X12_linear=TRUE){
  if (X12_linear){
    Z = qnorm(X[,1:2])
    kz1 = rnorm(nrow(X),mean=r*Z[,2], sd = sqrt(1-r^2))
    kz2 = rnorm(nrow(X),mean=r*Z[,1], sd = sqrt(1-r^2))
    return( cbind( pnorm(kz1),pnorm(kz2), matrix(runif(nrow(X)*(ncol(X)-2)),nrow(X),ncol(X)-2) ) )
  }else{# X[,1:2] generated with gen_2_vars_not_normal(n,r)
    # This is an approximation to the conditional distributions
    x = 2*X[,1]-1
    y = 2*X[,2]-1
    yy <- sign(x)*runif(nrow(X),sqrt(pmax(0,r^2-x^2)),sqrt(1-x^2))
    xx <- sign(y)*runif(nrow(X),sqrt(pmax(0,r^2-y^2)),sqrt(1-y^2))
    return( cbind( .5 + .5*xx, .5 + .5*yy, matrix(runif(nrow(X)*(ncol(X)-2)),nrow(X),ncol(X)-2) ) )
  }
}
