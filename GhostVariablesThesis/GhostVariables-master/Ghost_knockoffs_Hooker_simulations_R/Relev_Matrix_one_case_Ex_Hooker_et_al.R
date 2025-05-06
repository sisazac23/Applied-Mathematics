# Relevance Matrix for one dataset coming from 
# the simulation study
# "My_simul_Ex_Hooker_et_al.R"
#
##################
# Recreating the simulated example in Hooker et al. (2021)
source("aux_functs_simul_Ex_Hooker_et_al.R")
set.seed(1234)
#
# Generating model
# (x1,x2)~GaussianCopula(rho), x3...x10 iid U([0,1]), epsilon~N(0,s.eps^2)
# y = x1 + x2 + x3 + x4 + x5 + 0*x6 + 0.5*x7 + 0.8*x8 + 1.2*x9 + 1.5*x10 + epsilon
# 
# Data generating model parameters: 
X12_linear <- TRUE # FALSE # When FALSE, gen_2_vars_not_normal is used
rs = 0.9 # c(0,0.9) # c(.9,0.99) # c(0, .3, .6, .9) #  c(0.3,0.6) # vector of values of rho=cor(X_1,X_2)

s.eps = 0.1 # std. dev. of the noise epsilon
# ratio var(noise)/var(signal):
# s.eps=.1 => ratio=0.01  # Hooker et al simulations
# s.eps=.33=> ratio=0.1
# s.eps=.5 => ratio=0.2

# simulation parameters:
nsim = 1#50   # number of repetitions # Hooker et al simulations: 50
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
do_loco <- FALSE # TRUE # 
time.perm <- 0
time.cond <- 0
time.loco <- 0
time.gh <- 0
time.knoc <- 0

n = ns[1]
nt = nts[1]
r = rs[1]

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
  MSPEt.lin = mean( (yt - yt.hat)^2 )
  linmod = t.mod
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.lin)
    Imps.perm.lin = VI.perm
  })
  time.cond = time.cond + system.time({
    VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.lin)
    Imps.cond.lin = VI.cond
  })
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
  time.cond = time.cond + system.time({
    VI.cond = Vimp.perturb(t.mod,Xt,yt,pertXt=cXt,MSPEt.rf)
    Imps.cond.rf = VI.cond
  })
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
  nnmod = t.mod
  time.perm = time.perm + system.time({
    VI.perm = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=pXt,MSPEt.nn)
    Imps.perm.nn = VI.perm
  })
  time.cond = time.cond + system.time({
    VI.cond = Vimp.perturb(t.mod,data.frame(Xt),yt,pertXt=cXt,MSPEt.nn)
    Imps.cond.nn = VI.cond
  })
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

print(times.VI.methods)
print(times_models)

# Now lets look at some of these: means and standard deviations of importances
if (fit.lin){
  Imps.perm.lin.r = rank(Imps.perm.lin)
  Imps.cond.lin.r = rank(Imps.cond.lin)
  #Imps.loco.lin.r = rank(Imps.loco.lin)
  rel.gh.lin.r = rank(rel.gh.lin)
  rel.gh.lin.e.r = rank(rel.gh.lin.e)
  Imps.knoc.lin.r = rank(Imps.knoc.lin)
}
if (fit.rf){
  Imps.perm.rf.r = rank(Imps.perm.rf)
  Imps.cond.rf.r = rank(Imps.cond.rf)
  #Imps.loco.rf.r = rank(Imps.loco.rf)
  rel.gh.rf.r =    rank(rel.gh.rf,1:2)
  rel.gh.rf.e.r =  rank(rel.gh.rf.e,1:2)
  Imps.knoc.rf.r = rank(Imps.knoc.rf)
}
if (fit.nn){
  Imps.perm.nn.r = rank(Imps.perm.nn)
  Imps.cond.nn.r = rank(Imps.cond.nn)
  #Imps.loco.nn.r = rank(Imps.loco.nn)
  rel.gh.nn.r = rank(rel.gh.nn)
  rel.gh.nn.e.r = rank(rel.gh.nn.e)
  Imps.knoc.nn.r = rank(Imps.knoc.nn)
}


printing = FALSE#TRUE
# Now let's look at plots of linear model lm() for different methods
if (fit.lin){
  if (printing) pdf("one_case_lm_OLS_R.pdf", width = 10, height=5)
    par(mar=c(5,5,4,1))
    plot(Imps.perm.lin.r,type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    lines(Imps.cond.lin.r,type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.lin.r,type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.lin.e.r,type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.lin.r,type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.lin.r,type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
    title(bquote('Linear model fitted by OLS.'~rho==.(rs)))
  if (printing) dev.off()
} 

# Now let's look at plots of RF for different methods
if (fit.rf){
  if (printing) pdf("one_case_rf_R.pdf", width = 10, height=5)
    par(mar=c(5,5,4,1))
    plot(Imps.perm.rf.r,type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    lines(Imps.cond.rf.r,type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.rf.r,type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.rf.e.r,type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.rf.r,type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.rf.r,type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
    title(bquote('Random Forest.'~rho==.(rs)))
  if (printing) dev.off()
}

# Now let's look at plots of NNet for different methods
if (fit.nn){
  if (printing) pdf("one_case_nn_R.pdf", width = 10, height=5)
    par(mar=c(5,5,4,1))
    plot(Imps.perm.nn.r,type='b',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
    lines(Imps.cond.nn.r,type='b',col=2,pch=2,cex=1.5,lwd=2)
    if (do_loco) lines(Imps.loco.nn.r,type='b',col=3,pch=3,cex=1.5,lwd=2)
    lines(rel.gh.nn.e.r,type='b',col=4,pch=4,cex=1.5,lwd=2)
    lines(rel.gh.nn.r,type='b',col=5,pch=5,cex=1.5,lwd=2)
    lines(Imps.knoc.nn.r,type='b',col=6,pch=6,cex=1.5,lwd=2)
    legend('bottomright',c('P','C','L','G','Gp','Nk'),pch=1:6,col=1:6,cex=1,lwd=2)
    title(bquote('Neural Network.'~rho==.(rs)))
  if (printing) dev.off()
}

############ Relevance matrix
# variable relevance matrix
library(mgcv)
library(ggplot2)
library(grid)
library(maptools)# For pointLabel
source("../relev.ghost.var.R")
source("plot.relev.ghost.var.4.x.3.R")
#source("../relev.rand.perm.R")


printing = FALSE # TRUE # FALSE #

model.relev <- linmod
newdata <- data.frame(y=yt,Xt) 
file_pdf <- "Rel_Mat_one_case_lm.pdf"
relev.ghost.out <- relev.ghost.var(model=model.relev, 
                                   newdata = newdata,
                                   func.model.ghost.var= lm)
if (printing) pdf(file_pdf,height = 9, width = 9)
plot.relev.ghost.var.4.x.3(relev.ghost.out, n1=n)#, ncols.plot = 5)
#if (printing) pdf(file_pdf,height = 8, width = 16)
#plot.Matrix.relev.ghost.var(relev.ghost.out, n1=n, ncols.plot = 5)
if (printing) dev.off()

# summary(linmod)
# Call:
#   lm(formula = f, data = simdat)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -0.31107 -0.06786  0.00145  0.06446  0.32614 
# 
# Coefficients:
#                 Estimate  Std.Error t value Pr(>|t|)    
#   (Intercept)  0.0006166  0.0121059   0.051    0.959    
#   X1           0.9994076  0.0170830  58.503   <2e-16 ***
#   X2           0.9930606  0.0170144  58.366   <2e-16 ***
#   X3           1.0018519  0.0077832 128.720   <2e-16 ***
#   X4           0.9916691  0.0078429 126.441   <2e-16 ***
#   X5           1.0031305  0.0078463 127.847   <2e-16 ***
#   X6          -0.0025115  0.0078913  -0.318    0.750    
#   X7           0.5036489  0.0077832  64.709   <2e-16 ***
#   X8           0.7974914  0.0078910 101.064   <2e-16 ***
#   X9           1.2099143  0.0077681 155.754   <2e-16 ***
#   X10          1.5068578  0.0078163 192.783   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 0.1001 on 1989 degrees of freedom
# Multiple R-squared:  0.9896,	Adjusted R-squared:  0.9896 
# F-statistic: 1.898e+04 on 10 and 1989 DF,  p-value: < 2.2e-16

# 1-sum(linmod$residuals^2)/sum((y-mean(y))^2)
# [1] 0.989631


model.relev <- rfmod 
newdata <- data.frame(Xt)
file_pdf <- "Rel_Mat_one_case_rf.pdf"
relev.ghost.out <- relev.ghost.var(model=model.relev, y.ts=yt,
                                   newdata = Xt,
                                   func.model.ghost.var= lm)
if (printing) pdf(file_pdf,height = 9, width = 9)
plot.relev.ghost.var.4.x.3(relev.ghost.out, n1=n)#, ncols.plot = 5)
#if (printing) pdf(file_pdf,height = 8, width = 16)
#plot.Matrix.relev.ghost.var(relev.ghost.out, n1=n, ncols.plot = 5)
if (printing) dev.off()

#rfmod
# 
# Call:
#   randomForest(x = X, y = y) 
# Type of random forest: regression
# Number of trees: 500
# No. of variables tried at each split: 3
# 
# Mean of squared residuals: 0.1127781
# % Var explained: 88.26

# > sum((y-nnmod$fitted.values)^2)
# [1] 18.32606
# > sum((y-rfmod$predicted)^2)
# [1] 225.5561
# > 1-sum((y-rfmod$predicted)^2)/sum((y-mean(y))^2)
# [1] 0.8826064

model.relev <- nnmod 
newdata <- data.frame(y=yt,Xt) 
file_pdf <- "Rel_Mat_one_case_nn.pdf"
relev.ghost.out <- relev.ghost.var(model=model.relev, 
                                   newdata = newdata,
                                   func.model.ghost.var= lm)
if (printing) pdf(file_pdf,height = 9, width = 9)
plot.relev.ghost.var.4.x.3(relev.ghost.out, n1=n)#, ncols.plot = 5)
#if (printing) pdf(file_pdf,height = 8, width = 16)
#plot.Matrix.relev.ghost.var(relev.ghost.out, n1=n, ncols.plot = 5)
if (printing) dev.off()

#sum((y-nnmod$fitted.values)^2)
#[1] 18.32606
#sum(nnmod$residuals^2)
#[1] 18.32606
#1-sum(nnmod$residuals^2)/sum((y-mean(y))^2)
#[1] 0.990462

