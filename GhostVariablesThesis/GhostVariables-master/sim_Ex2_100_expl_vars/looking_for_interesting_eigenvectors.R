# 2 June 2022
V <- relev.ghost.lm.out$V
use.lasso <- FALSE # TRUE # FALSE # 
if (use.lasso){
  V <- relev.ghost.lasso.out$V
  zero.coefs <- which(diag(V)==0)
  V <- V[-zero.coefs,-zero.coefs]
}
eigV <- eigen(V)
vals <- eigV$values
vecs <- eigV$vectors

p<-length(vals)

par(mfrow=c(3,4))
for (j in c(1:6,p-(5:0))){
  plot(vecs[,j],main=j)
  abline(h=0,col=8)
  abline(h=median(vecs[,j]),col=2)
  abline(h=median(vecs[,j])+6*c(-1,1)*mad(vecs[,j]),col=4,lty=2)
}
par(op)
par(mfrow=c(1,1))

# 3 June 2022

plot(vals)
plot(-diff(log(vals)));abline(h=0,col=8)
boxplot(-diff(log(vals)));abline(h=0,col=8)
bp.aux <- boxplot(log(-diff(log(vals))))

which( log(-diff(log(vals))) > bp.aux$stats[5])

scale.med <- function(x,constant=1.4826){(x-median(x))/mad(x,constant = constant)}

par(mfrow=c(3,4))
for (j in c(1:6,p-(5:0))){
  svj <- scale.med(vecs[,j])
  plot(svj,main=paste0("j= ",j,";   amplitude= ",round(diff(range(svj)))))
  abline(h=0,col=8)
#  abline(h=median(vecs[,j]),col=2)
#  abline(h=median(vecs[,j])+6*c(-1,1)*mad(vecs[,j]),col=4,lty=2)
}
par(op)

####
matplot(vecs,type="l");abline(h=0,col=2)
matplot(vecs^2,type="l");abline(h=0,col=2)

par(mfrow=c(3,4))
for (j in 1:p){
  plot(vecs[,j],main=paste0(j,";  ",signif(vals[j],digits = 2)))
  abline(h=0,col=2)
}
par(mfrow=c(1,1))

boxplot(vecs)
boxplot(apply(vecs,2,scale)) 

plot(apply(apply(vecs,2,scale)^4,2,mean))
plot(apply(apply(vecs^2,2,scale)^4,2,mean))

plot(apply(apply(vecs,2,scale.med)^4,2,mean))# Good detection! kurtosis coefficient
plot(apply(apply(vecs^2,2,scale.med)^4,2,mean))

matplot(apply(vecs,2,scale),type="l")

matplot(apply(vecs,2,scale.med),type="l")# Good detection!

boxplot(apply(vecs,2,scale.med),type="l")
boxplot(apply(vecs,2,scale.med)^2,type="l")
boxplot(log(apply(vecs,2,scale.med)^2),type="l")

plot(apply(apply(vecs,2,scale.med)^2,2,mean))# Good detection!

boxplot(apply(apply(vecs,2,scale.med)^2,2,mean))
boxplot(log(apply(apply(vecs,2,scale.med)^2,2,mean)))

boxplot(apply(apply(vecs,2,scale.med),2,mean))
boxplot(apply(apply(vecs,2,scale.med),2,var))

boxplot(apply(apply(vecs,2,scale.med),2,var))$out
boxplot(log(apply(apply(vecs,2,scale.med)^2,2,mean)))$out

boxplot(apply(apply(vecs,2,scale.med)^2,2,mean))$out

mean.scl.med <- apply(apply(vecs,2,scale.med)^2,2,mean)
border <- boxplot(apply(apply(vecs,2,scale.med)^2,2,mean))$stats[5]
which(mean.scl.med>border)
interesting_eig_vec <- which(mean.scl.med>border)
matplot(apply(vecs,2,scale.med)[,interesting_eig_vec],type="l")

par(mfrow=c(3,4))
for (j in interesting_eig_vec){plot(apply(vecs,2,scale.med)[,j],type="l",main=j);abline(h=0,col=2)}

par(mfrow=c(3,4))
for (j in interesting_eig_vec){plot(sort(apply(vecs,2,scale.med)[,j]),type="l",main=j);abline(h=0,col=2)}

par(mfrow=c(3,4))
for (j in interesting_eig_vec){hist(apply(vecs,2,scale.med)[,j],main=j,breaks = 10);abline(h=0,col=2)}

par(mfrow=c(1,1))

apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,mean)
apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,var)
apply(apply(vecs^2,2,scale.med)[,interesting_eig_vec],2,mean)

plot(apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,var))
plot(log(apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,var)))

apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,sd)
plot(apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,sd))
plot(log(apply(apply(vecs,2,scale.med)[,interesting_eig_vec],2,sd)))

apply(apply(vecs,2,scale.med)[,interesting_eig_vec]^4,2,mean)
plot(apply(apply(vecs,2,scale.med)[,interesting_eig_vec]^4,2,mean))
plot(log(apply(apply(vecs,2,scale.med)[,interesting_eig_vec]^4,2,mean)))
