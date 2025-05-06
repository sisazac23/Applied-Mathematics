# Figure Relevance matrix for one case of Example 2
#
load("output_simulation/one_case_Ex3_seed_123456.Rdata")
# load("output_simulation/one_case_Ex3_seed_13579.Rdata")

printing = FALSE#TRUE
if (1==1){ 
  if (printing) pdf("one_case_Ex2_rel_meas.pdf", width = 10, height=10)
  op <- par(mfrow=c(2,1))

  par(mar=c(5,5,4,1))
  plot(Imps.perm.lin.r,type='p',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lin.r,type='p',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lin.r,type='p',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lin.r,type='p',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lin.e.r,type='p',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lin.r,type='p',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c('P','L','G','Gp','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1,lwd=2,bg="white")
  title(bquote('Linear model estimated by OLS'))

  par(mar=c(5,5,4,1))
  plot(Imps.perm.lasso.r,type='p',pch=1,cex=1.5,lwd=2,xlab='Features',ylab='Importance Rank',cex.lab=1.5,cex.axis=1.5,ylim=c(1,nfeat))
  #lines(Imps.cond.lasso.r,type='p',col=2,pch=2,cex=1.5,lwd=2)
  if (do_loco) lines(Imps.loco.lasso.r,type='p',col=3,pch=3,cex=1.5,lwd=2)
  lines(rel.gh.lasso.r,type='p',col=4,pch=4,cex=1.5,lwd=2)
  lines(rel.gh.lasso.e.r,type='p',col=5,pch=5,cex=1.5,lwd=2)
  lines(Imps.knoc.lasso.r,type='p',col=6,pch=6,cex=1.5,lwd=2)
  legend('topright',c('P','L','G','Gp','Nk'),
         pch=c(1,3:6),col=c(1,3:6),cex=1,lwd=2,bg="white")
  title(bquote('Linear model estimated by lasso'))
  par(op)  
  if (printing) dev.off()
}

################################################

if (1==1){ 
  if (printing) pdf("one_case_Ex2_rel_Gh.pdf", width = 10, height=5)
  op <- par(mfrow=c(1,1))
  par(mar=c(4,5,3,1))
  plot(rel.gh.lin.r,type='p', pch=24, cex=1, lwd=2, col="blue",
       xlab='Features',ylab='Importance Rank',
       main='Linear model. Relevance by ghost variables',
       cex.lab=1.5,cex.main=1.5,cex.axis=1,
       ylim=c(1,nfeat))
  lines(rel.gh.lasso.r,type='p',pch=25,cex=1,lwd=2, col="red")
  lines(rel.gh.lin.r,type='p',pch=24,cex=1,lwd=2, col="blue")
  legend('topright',c('OLS','Lasso'),
         pch=c(24,25),col=c('blue','red'),cex=1,bg="white")
  par(op)  
  if (printing) dev.off()
}

################################################

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


plot(vals,ylab="Eigenvalues",main="Eigenvalues of the relevance matrix V")

plot(-diff(vals));abline(h=0,col=8)
bp.aux <- boxplot(log(-diff(vals)))

(aux <- which( log(-diff(vals)) > bp.aux$stats[5]))
#[1]  1  2  3  4  5  6 99

(select_eig_vals <- sort(union(aux,aux+1)))
#[1]   1   2   3   4   5   6   7  99 100

if (1==1){
  if (printing) pdf("one_case_Ex2_eig_vals.pdf", width =10, height=5)
  par(mfrow=c(1,1))
  par(mar=c(4,4.5,3,1))
  plot(vals,cex.lab=1.5,cex.main=1.5,
       ylab="Eigenvalues",main="OLS estimation. Eigenvalues of the relevance matrix V")
  lines(select_eig_vals[1:7],vals[select_eig_vals[1:7]],lwd=2)
  lines(select_eig_vals[8:9],vals[select_eig_vals[8:9]],lwd=2)
  points(select_eig_vals,vals[select_eig_vals],pch=21,col=1,bg=1,cex=1.2)
  legend("topright","Eigenvalues around large steps",pch=21,col=1,pt.bg=1,pt.cex=1.2,bty="n")
  if (printing) dev.off()
}

#scale.med <- function(x,constant=1.4826){(x-median(x))/mad(x,constant = constant)}

if (1==1){
  if (printing) pdf("one_case_Ex2_eig_vects.pdf", width = 8, height=6)
  par(mfrow=c(3,3))
  par(mar=c(4,4,3,1))
  for (j in select_eig_vals){
    #svj <- scale.med(vecs[,j])
    svj <- vecs[,j]
    plot(svj, xlab="Variables", ylab="Loadings",
         main=paste0("Eigenvector ",j,". Expl.Var.=",round(100*vals[j]/sum(vals),2),"%"))
    abline(h=0,col=8)
    #  abline(h=median(vecs[,j]),col=2)
    #  abline(h=median(vecs[,j])+6*c(-1,1)*mad(vecs[,j]),col=4,lty=2)
  }
  par(op)
  if (printing) dev.off()
}



