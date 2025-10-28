library(tidyverse)
library(gtools)
library(impute)
library(dplyr)
library(outliers)
library(sva)
library(reshape)
library(ggbiplot)
library(gridExtra)

rm(list=ls())
setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')

load("data/metabolome/ProcessedData/normalizedData")

par(mfrow=c(2,2))
## adjust for technical covariates (pre-LCMS variables)
hist(dat$elapse_DOC_LOG) # travel time from collection location to Texas A&M
hist(dat$serum_temp) # sample temperature at Texas A&M
plot( serum_temp ~ elapse_DOC_LOG, dat, las=1)
plot( graded_hemolysis ~ elapse_DOC_LOG, dat, las=1)
cor.test(dat$graded_hemolysis, dat$elapse_DOC_LOG, method= 'spear')

plot( graded_hemolysis ~ serum_temp, dat, las=1)
cor.test(dat$graded_hemolysis, dat$serum_temp , method= 'spear')

table(dat$graded_hemolysis)
dat$graded_hemolysis <- as.factor(dat$graded_hemolysis)
summary(ordinal::clm(graded_hemolysis ~ serum_temp, data=dat)) # 
# neither serum temperature, nor shipment time associated significantly with hemolysis


## get an idea of the issues:
p <- list()
for(i in 1:length(dogmzs)){
s <- summary(lm(dat[ ,dogmzs[i]] ~ elapse_DOC_LOG + serum_temp + as.numeric(graded_hemolysis), dat, na.action=na.exclude))
p[[i]] <- s$coefficients[2:4,4] }
p <- as.data.frame(do.call(rbind, p))
rownames(p) <- dogmzs


p[order(p$elapse_DOC_LOG), ][1:10, ]

# plot some offenders
par(mfrow=c(2,2))
plot(Choline ~ elapse_DOC_LOG, dat, pch=19, cex=0.5)
plot(Choline ~ serum_temp, dat, pch=19, cex=0.5) 
plot(Glucoronate ~ graded_hemolysis, dat, pch=19, cex=0.5)

colSums(is.na(dat[ ,dogmzs]))

tech <- dat[ ,dogmzs]

for(i in 1:length(dogmzs)){
  tech[ ,dogmzs[i]] <- residuals(lm(tech[ ,dogmzs[i]] ~ elapse_DOC_LOG + serum_temp + as.numeric(graded_hemolysis), dat, na.action=na.exclude)) } # 6 samples with NA for serum temp

dat[!is.na(dat$serum_temp) ,dogmzs] <- tech[!is.na(dat$serum_temp), ]

colSums(is.na(dat[ ,dogmzs]))

# plot the (reformed) offenders
plot(Choline ~ elapse_DOC_LOG, dat, pch=19, cex=0.5) 
plot(Choline ~ serum_temp, dat, pch=19, cex=0.5) 
plot(Glucoronate ~ graded_hemolysis, dat, pch=19, cex=0.5)


save(dat, dogmzs, file="data/metabolome/ProcessedData/technicalCovarsRemovedData")
write.table(file='data/metabolome/ProcessedData/technicalCovarsRemoved.txt', dat) # for collaborators


