plot.relev.ghost.var.4.x.3 <- function(relev.ghost.out, n1,
                                 vars=NULL, sum.lm.tr=NULL,
                                 alpha=.01, ncols.plot=3){
  A <- relev.ghost.out$A
  V <- relev.ghost.out$V
  eig.V <- relev.ghost.out$eig.V
  GhostX <- relev.ghost.out$GhostX
  relev.ghost <- relev.ghost.out$relev.ghost
  resid.var <- relev.ghost.out$MSPE.test
  
  p  <- dim(A)[2]
  
  ncols.plot<-3 
  max.plots <- 4*ncols.plot
  if (is.null(vars)){
    vars <- 1:min(max.plots,p)
  }else{
    if (length(vars)>max.plots){
      vars <- vars[1,max.plots]
      warning(
        paste("Only the first", max.plots, "selected variables in 'vars' are used"))
    }
  }
  n.vars <- length(vars)
  #nrows.plot <- 1 + n.vars%/%ncols.plot + (n.vars%%ncols.plot>0)
  nrows.plot <- n.vars%/%ncols.plot + (n.vars%%ncols.plot>0) # It must be equal to 4
  
  if (!is.null(sum.lm.tr)){
    #  F.transformed <- resid.var*sum.lm.tr$coefficients[-1,3]^2/n1
    F.transformed <- sum.lm.tr$coefficients[-1,3]^2/n1
  }
  # F.critic.transformed <- resid.var*qf(1-alpha,1,n1-p-1)/n1
  F.critic.transformed <- qf(1-alpha,1,n1-p-1)/n1
  
  rel.Gh <- data.frame(relev.ghost=relev.ghost)
  rel.Gh$var.names <- colnames(A)
  
  if (!is.null(sum.lm.tr)){
    plot.rel.Gh <- ggplot(rel.Gh) +
      geom_bar(aes(x=reorder(var.names,X=length(var.names):1), y=relev.ghost), 
               stat="identity", fill="darkgray") +
      ggtitle("Relev. by ghost variables") +
      geom_hline(aes(yintercept = F.critic.transformed),color="blue",size=1.5,linetype=2)+
      theme(axis.title=element_blank())+
      theme_bw()+
      ylab("Relevance")+
      xlab("Variable name") +
      coord_flip()
  }else{
    plot.rel.Gh <- ggplot(rel.Gh) +
      geom_bar(aes(x=reorder(var.names,X=length(var.names):1), y=relev.ghost), 
               stat="identity", fill="darkgray") +
      ggtitle("Relev. by ghost variables") +
      #geom_hline(aes(yintercept = F.critic.transformed),color="blue",size=1.5,linetype=2)+
      theme(axis.title=element_blank())+
      theme_bw()+
      ylab("Relevance")+
      xlab("Variable name") +
      coord_flip()
  }
  
  plot.rel.Gh.pctg <- ggplot(rel.Gh) +
    geom_bar(aes(x=reorder(var.names,X=length(var.names):1), 
                 y=100*relev.ghost/sum(relev.ghost)), 
             stat="identity", fill="darkgray") +
    coord_flip() +
    ggtitle("Relev. by ghost variables (% of total relevance)") +
    theme(axis.title=element_blank())
  
  # eigen-structure
  # eig.V <- eigen(V)
  eig.vals.V <- eig.V$values
  eig.vecs.V <- eig.V$vectors
  
  expl.var <- round(100*eig.vals.V/sum(eig.vals.V),2)
  cum.expl.var <- cumsum(expl.var)
  
  # op <-par(mfrow=c(2,2))
  # plot(eig.vals.V, main="Eigenvalues of matrix V",ylab="Eigenvalues", type="b")
  # for (j in (1:p)){
  #   plot(eig.V$vectors[,j],main=paste("Eigenvector",j,", Expl.Var.:",expl.var[j],"%"))
  #   abline(h=0,col=2,lty=2)
  # }
  # par(op)
  
  
  eig.V.df <- as.data.frame(eig.V$vectors)
  names(eig.V.df) <- colnames(A)
  
  op <-par(mfrow=c(nrows.plot,ncols.plot))
  plot(0,0,type="n",axes=FALSE,xlab="",ylab="")
  
  # if (!is.null(sum.lm.tr)){
  #   plot(F.transformed,relev.ghost,
  #        xlim=c(0,max(c(F.transformed,relev.ghost))),
  #        ylim=c(0,max(c(F.transformed,relev.ghost))),
  #        #         xlab=expression(paste("F-statistics*",hat(sigma)^2/n[1])), 
  #        xlab=expression(paste("F-statistics/",n[1])), 
  #        ylab="Relev. by ghost variables")
  #   pointLabel(F.transformed,relev.ghost, colnames(A))
  #   abline(a=0,b=1,col=2)
  #   abline(v=F.critic.transformed,h=F.critic.transformed,lty=2,col="blue",lwd=2)
  # }else{
  #   plot(0,0,type="n",axes=FALSE,xlab="",ylab="")
  # } 
  
  par(xaxp=c(1,p,min(p,5)))
  plot(eig.vals.V, ylim=c(0,max(eig.vals.V)),
       main=expression("Eigenvalues of matrix V"),
       ylab="Eigenvalues",type="b")
  abline(h=0,col="red",lty=2)
  
  par(op)
  
  pushViewport(viewport(layout = grid.layout(nrows.plot, ncols.plot))) #package grid
  vplayout <- function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)
  print(plot.rel.Gh,vp = vplayout(1,1))

  #First eigenvector plotted in the last column of the first row
  j <- vars[1]
  print(
    ggplot(eig.V.df) +
      #       geom_bar(aes(x=var.names, y=eig.V.df[,j]),
      geom_bar(aes(x=reorder(names(eig.V.df),X=length(names(eig.V.df)):1), 
                   y=eig.V.df[,j]), stat="identity") +
      geom_hline(aes(yintercept=0),color="red",linetype=2,size=1) +
      #ylim(min(eig.V.df[,j])-.5,max(eig.V.df[,j])+.5) +
      coord_flip() +
      ggtitle(paste0("Eig.vect.",j,", Expl.Var.: ",expl.var[j],"%")) +
      theme(axis.title=element_blank(),plot.title = element_text(size = 12)),
    #vp = vplayout(2+(jj-1)%/%ncols.plot, 1+(jj-1)%%ncols.plot)
    vp = vplayout(1,3)
  )
  
  #Other eigenvectors plotted from row 2
  jj<-0
  for (j in vars[-1]){
    jj<-jj+1
    print(
      ggplot(eig.V.df) +
        #       geom_bar(aes(x=var.names, y=eig.V.df[,j]),
        geom_bar(aes(x=reorder(names(eig.V.df),X=length(names(eig.V.df)):1), 
                     y=eig.V.df[,j]), stat="identity") +
        geom_hline(aes(yintercept=0),color="red",linetype=2,size=1) +
        #ylim(min(eig.V.df[,j])-.5,max(eig.V.df[,j])+.5) +
        coord_flip() +
        ggtitle(paste0("Eig.vect.",j,", Expl.Var.: ",expl.var[j],"%")) +
        theme(axis.title=element_blank(),plot.title = element_text(size = 12)),
      vp = vplayout(2+(jj-1)%/%ncols.plot, 1+(jj-1)%%ncols.plot)
    )
  }
}
