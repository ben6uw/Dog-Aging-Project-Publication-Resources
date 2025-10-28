library(tidyverse)
library(ggplot2)
library(dplyr)
library(ggpubr)


setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')

rm(list=ls())
################################################################################################
## How much of each term is reduced/affected by the presence of the GRM?
#################################################################################################
load('data/metabolome/ProcessedData/P1.metabolome.withBreedData.2') # see 'signalment.R
load('dog metabolome paper/PCANOVA_result') # see 'signalment.R'
load('data/scaled.data_for_CBC_mixedModel')

table(dat$commonPurebred)

pca <- prcomp(dat[ ,dogmzs], scale=T)
eigs <- pca$sdev^2
o <- AssocTests::tw(eigs, eigenL=length(eigs), criticalpoint=0.9793) # alpha at 5%
sigEigs <- print(o$SigntEigenL)
PCs <- as.data.frame(pca$x[ ,1:sigEigs])
pcs <- colnames(PCs)
PCs$dog_id <- dat$dog_id

propVar <- print(summary(pca)$importance[2, 1:sigEigs]) # prop. var exp
sum(summary(pca)$importance[2, ][1:sigEigs]) # cumulative proportion of variance explained

dat$sqrtAge <- sqrt(dat$AgeAtDOC)
dat$sqrtWT <- sqrt(dat$weight_at_DOC)

pcs <- names(PCs)
dat <- merge(dat, PCs)
dat$commonPurebred <- as.factor(dat$commonPurebred)
dat$lifestage_at_DOC <- factor(dat$lifestage_at_DOC, levels=c('Puppy', 'Young', 'Mature', 'Senior'))

table(dat$dog_id %in% rownames(grm))
dat <- dat[dat$dog_id %in% rownames(grm), ] # dat is now composed of variables that have been scaled
grm <- grm[dat$dog_id, dat$dog_id] # subset and order the GRM by the dogs in the mz data

###################################

Zmat <- diag(nrow(dat)) # an empty design matrix for the random effects

# build a design matrix for the fixed effects; specify interaction terms here
terms
cat(paste0('+ ', terms))# manually copy and paste this into the line below.

X <- model.matrix( ~ sqrtAge + sqrtWT + lifestage_at_DOC + sex + sterilization_status + hours_fasting + commonPurebred + krt_cbc_total_wbcs + krt_cbc_abs_bands + krt_cbc_abs_neutrophils + krt_cbc_abs_lymphocytes + krt_cbc_abs_monocytes + krt_cbc_abs_eosinophils + krt_cbc_abs_basophils + krt_cbc_rbc + krt_cbc_hgb + krt_cbc_hct + krt_cbc_mcv + krt_cbc_mch + krt_cbc_mchc + krt_cbc_rdw + krt_cbc_mpv + krt_cbc_pct + krt_cbc_retic_abs, data=dat) 

pcVars <- paste0('PC', 1:sigEigs)
pcDat <- dat[ ,pcVars]

## make a null GRM, equivant to an identity matrix, diag=1, off-diag=0
mean(grm)
nullGRM <-  matrix(0, nrow(grm), ncol(grm)) 
diag(nullGRM) <- 1

# nullGRM <- scale(nullGRM) # may want to scale the null matrix to avoid convergence problems in REML 
nullGRM[1:4,1:4]

library(parallel)
library(doParallel)
library(EMMREML)

n.cores <- detectCores(all.tests = F, logical = T) # detects the number of available cores
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  

emma <- emmreml(y=pcDat[ ,1], X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) 
emmaNULL <- emmreml(y=pcDat[ ,1], X=X, Z=Zmat, K=nullGRM, varbetahat = T, varuhat=T, PEVuhat=T, test=T) # test the null grm for ML failure

clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  
clusterExport(clus, varlist=c("pcDat", 'pcVars', 'X', 'Zmat', 'grm'), envir = environment()) 

mList <- list()
nullmList <- list()

for(i in 1:length(pcVars)) {
  clusterExport(clus, varlist=c("pcDat", 'pcVars', 'X', 'Zmat', 'grm', 'nullGRM'), envir = environment()) 
  Y <- pcDat[ ,i]
  mList[[i]] <- emmreml(y=Y, X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) 
  nullmList[[i]] <- emmreml(y=Y, X=X, Z=Zmat, K=nullGRM, varbetahat = T, varuhat=T, PEVuhat=T, test=T)}


save(mList, nullmList, dat, pcVars, X, sigEigs, file='dog metabolome paper/effect_of_GRM_results')



rm(list=ls())
##########################################
load('data/metabolome/ProcessedData/P1.metabolome.withBreedData.2') # see 'signalment.R
load('dog metabolome paper/PCANOVA_result') # see 'signalment.R'
load('data/scaled.data_for_CBC_mixedModel')
load('dog metabolome paper/effect_of_GRM_results')

# [stretegy for getting the varaince explained, based on post from Ben Bolker: https://stackoverflow.com/questions/74187391/r-linear-mixed-effects-finding-individual-fixed-effects-variances-in]:
# Estimating sigma^2_f can, in principle, be carried out by predicting fitted values based on the fixed effects alone (equivalent to multiplying the design matrix of the fixed effects with the vector of fixed effect estimates) followed by calculating the variance of these fitted values (Snijders & Bolker 1999).

# if i divide the variance of the BLUES by the variance of the PC, this is the var exp.

i=1
plot(mList[[i]]$betahat ~ nullmList[[i]]$betahat)
abline(0,1)

m <- mList[[i]]
modelVars <- rownames(m$betahat)
breedVars <- modelVars[grepl('common', modelVars)]
breedVars

X[ ,breedVars] # design matrix of breed information (1, 0)
m$betahat[breedVars, ] # fixed effects of each breed

X[ ,breedVars] %*% m$betahat[breedVars, ] # matrix multipication, gets BLUES

plot(mList[[1]]$betahat, mList[[2]]$betahat)

grmB <- sapply(mList, function(x) X[ ,breedVars] %*% x$betahat[breedVars, ])
naiveB <- sapply(nullmList, function(x) X[ ,breedVars] %*% x$betahat[breedVars, ])

colnames(grmB) <- pcVars
colnames(naiveB) <- pcVars

grmV <- apply(grmB, 2, var)
naiveV <- apply(naiveB, 2, var)

variances <- data.frame(PC=pcVars, 'PCvariance'= apply(dat[ ,pcVars], 2, var), 'grm'=grmV, 'naive'=naiveV)
head(variances)
l <- pivot_longer(variances, -c(PC, PCvariance), names_to = 'model', values_to='variance')

l$PC <- factor(l$PC, levels=pcVars)
levels(l$PC) <- c(1:sigEigs)
l$model <- factor(l$model, levels=c('naive', 'grm'))

ggplot(l, aes(y=variance/PCvariance, x=PC, fill=model))+
  geom_bar(stat = 'identity', position='dodge')+
  theme_classic(base_size = 16)+
  scale_fill_manual(values=c('orange', 'purple'))+
  ylab('PC variance explained by 8 common breeds')


# this plot is misleading in the sense that the variance explained by the GRM is huge in comparison to the amount of variation by breed that it accounts for

# incorporating the GRM reduced the variance explaned by the 8 common breeds on some, but not all PCs.  
# so, it could be that non-genetic aspects of the 8 common breeds are affecting the plasma metabolome.

# Qs:
# what non-genetic aspect of dogs are they (the breeds) explaining?


# we could:
# 1) see the degree to which the presence of weight influences the effect of breed?
# 2) see what other terms are affected by the grm

modelVars <- rownames(m$betahat)
breedVars <- modelVars[grepl('common', modelVars)]
lifeStageVars <- modelVars[grepl('lifestage', modelVars)]
cbcVars <- modelVars[grepl('krt', modelVars)]
sigVars <- modelVars[!modelVars %in% c(breedVars, lifeStageVars, cbcVars)]



VarList <- list("sqrtAge", "sqrtWT", "sexMale", "sterilization_statussterile", "hours_fasting", lifeStageVars, breedVars, cbcVars)
     
m <- mList[[1]] # pull a model as an example

variance <- matrix(nr=16, nc=length(pcVars))
tmp <-matrix(nr=nrow(dat), ncol=8)

for(k in 1:length(pcVars)) {
  m <- mList[[k]] 
for(i in 1:length(VarList)){
  tmp[ ,i] <- if(length(VarList[[i]]) >1) { X[ ,VarList[[i]]] %*% m$betahat[VarList[[i]], ] } else{ X[ ,VarList[[i]]] * m$betahat[VarList[[i]], ]} }
variance[1:length(VarList), k] <- apply(tmp, 2, var)

for(i in 1:length(VarList)){
  m <- nullmList[[k]] 
  tmp[ ,i] <- if(length(VarList[[i]]) >1) { X[ ,VarList[[i]]] %*% m$betahat[VarList[[i]], ] } else{ X[ ,VarList[[i]]] * m$betahat[VarList[[i]], ]} }
variance[1:length(VarList) + length(VarList), k] <- apply(tmp, 2, var) }

variance <- as.data.frame(variance)
colnames(variance) <- pcVars

variance$term  <- factor(rep(c("age", "weight", "sex", "sterilization", "fasting", 'lifestage', 'breed', 'cbc'), 2), levels=c("age", "weight", "sex", "sterilization", "fasting", 'lifestage', 'breed', 'cbc'))

variance$model <- factor(c(rep('grm', 8), rep('naive', 8)), levels=c('naive', 'grm'))
head(variance) 

l <- pivot_longer(variance, -c(term, model), values_to = 'variance', names_to = 'PC')

l$PC <- factor(l$PC, levels=pcVars)
pcVariance <- apply(dat[ ,pcVars], 2, var)

l$pcVariance <- pcVariance[l$PC]
head(l)
l$propVar <- l$variance/l$pcVariance

ggplot(l, aes(y=propVar, x=term, color=term, fill=model))+
  geom_bar(stat = 'identity', position='dodge')+
  theme_classic(base_size = 16)+
  scale_fill_manual(values=c('white', 'grey40'))+
  ylab('prop of PC explained')+
  xlab('')+
  coord_flip()+
  facet_wrap(~PC)


# look at the % change in variance explaied +/- grm:
head(l)
w <- pivot_wider(l[ ,c(1:3, 6)], names_from=model, values_from=propVar)
head(w)
w$pct.change <- ((w$grm-w$naive)/w$naive)*100
w$difference<- w$grm-w$naive
head(w)

meanVar <- w %>% group_by(term) %>% summarise_at(vars('naive', 'grm'), mean)
head(meanVar)
meanVar <- pivot_longer(meanVar, cols=c(naive, grm), names_to='model', values_to='meanVar')
meanVar$model <- factor(meanVar$model, levels=c('naive', 'grm'))

ggplot(meanVar, aes(y=meanVar, x=term, fill=model))+
  geom_bar(stat = 'identity', position='dodge')+
  theme_classic(base_size = 14)+
  scale_fill_manual(values=c('grey', 'purple'))

differenceFactetPlot <- ggplot(w, aes(y=difference, x=PC, fill=term))+
  geom_bar(stat = 'identity')+
  theme_classic(base_size = 12)+
  theme(axis.text.y=element_text(size=7))+
  theme(legend.position = 'none')+
  ylab('change in variance explained (V with GRM - V without GRM)')+
  xlab('')+
  coord_flip()+
  facet_wrap(~term)+
  scale_x_discrete(limits=rev)

range(w$pct.change[w$term=='breed'])


## limit to the pcs with breed effects (ANOVA P<5%)
load('dog metabolome paper/PCANOVA_result')
ANOVAbreedP <- sapply(aList, function(x) x$`Pr(>F)`[7])
breedPCs <- pcVars[ANOVAbreedP<=0.05] # PCS for which breed is signficant at P<=5%

head(w)
meanVar <- w[w$PC %in% breedPCs, ] %>% group_by(term) %>% summarise_at(vars('naive', 'grm'), mean)
head(meanVar)
meanVar$meanPCvarDiff <- (meanVar$grm - meanVar$naive) / meanVar$naive * 100 # % difference in mean PC var with GRM in the model

meanVar # these are the mean difference in var exp by models +/- the GRM


meanVar <- pivot_longer(meanVar, cols=c(naive, grm), names_to='model', values_to='meanVar')
meanVar$model <- factor(meanVar$model, levels=c('naive', 'grm'))
levels(meanVar$model) <- c('naive', '+GRM')

head(meanVar)

removeGRMplot <- ggplot(meanVar, aes(y=meanVar, x=term, fill=model))+
  geom_bar(stat = 'identity', position='dodge')+
  theme_classic(base_size = 18)+
  scale_fill_manual(values=c('grey', 'purple'))+
  ylab('average variance (5 breed PCs)')+
  xlab('')+
  scale_y_continuous(breaks = round(seq(min(0), max(0.06), by = 0.02) , 10))+
  theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust=1)) 

removeGRMplot  

save(removeGRMplot, file='dog metabolome paper/removeGRMplot')

