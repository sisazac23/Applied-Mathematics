# graphic for the paper in TEST
library(ggplot2)
library(grid)
library(maptools)# For pointLabel

source("plot.relev.ghost.var.4.x.3.R")

load("nnet_addendum_2021_07_19.RData")

# sort(relev.ghost.out$relev.ghost, decreasing = TRUE)
#       log.size     categ.distr     type.chalet       Barcelona 
#    0.456746940     0.198457626     0.106994543     0.074406747 
#      bathrooms  log_activation           floor           rooms 
#    0.071401359     0.055728077     0.038746278     0.034168191 
#    type.studio     type.duplex         hasLift  type.penthouse 
#    0.031968985     0.011604683     0.010718331     0.010032464 
# ParkingInPrice        exterior hasParkingSpace       floorLift 
#    0.004874936     0.003056064     0.001938931     0.001447943 

#pdf(file="VarRlevIdealista_nn_Gh_3_3_2022.pdf", height=16, width=12)
plot.relev.ghost.var.4.x.3(relev.ghost.out, n1=n.tr, 
                           vars=1:10, sum.lm.tr=NULL,
                           alpha=.01, ncols.plot=3)
#dev.off()

#pdf(file="VarRlevIdealista_nn_Gh_3_3_2022_7_16.pdf", height=16, width=12)
plot.relev.ghost.var.4.x.3(relev.ghost.out, n1=n.tr, 
                           vars=7:16, sum.lm.tr=NULL,
                           alpha=.01, ncols.plot=3)
#dev.off()

