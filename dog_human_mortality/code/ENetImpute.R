library(tidyverse)
library(gtools)
library(impute)
library(dplyr)
library(outliers)
library(sva)
library(reshape)
library(ggbiplot)
library(gridExtra)
library(ggpubr)
library(readxl)
library(pheatmap)
library(missForest)
library(caret)



# Elastic Net imputation 

rm(list=ls())

load('data/metabolome/RawData/joinedNormDataMissRemovedBatchOrderAdjusted')
adjDat[1:5,1:10]
adjDat <- adjDat[ ,c("Sample.ID", "prep_batch", "dog_id", "dog_id_cohort", "run", dogmzs)]

hist(colSums(is.na(adjDat[ ,dogmzs])), border=0, col='grey40', xlab='missing (n=1,880 samples)', ylab='metabolites (n=137)')

missingNess <- data.frame('metabolite'=dogmzs, 'Nmissing'=colSums(is.na(adjDat[ ,dogmzs])), 'Total.Samples'=1880)

save(missingNess, file='data/metabolome/missingNess')



imputeUS <- dogmzs[colSums(is.na(adjDat[ ,dogmzs]))>0] # metabolites to impute


cores <- detectCores() 
cl <- makeCluster(cores)

registerDoParallel(cl)
Ximp <- missForest(adjDat[ ,dogmzs], parallelize= 'variables')$ximp # impute all possible predictors (all mzs)

mzdat <- adjDat[ ,dogmzs]
ENimputed <- adjDat[ ,dogmzs] # will replace with imputed data

fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 10)
ENlist <- list() # save predictor models in case we want to look at accuracy, or make addtional predictions

for(k in 1:length(imputeUS)){
  Y <- select(mzdat, imputeUS[k])
  X <- select(Ximp, -imputeUS[k])
  tmpDat <- cbind(Y, X)
  trainDat <- tmpDat[!is.na(tmpDat [ ,1]), ] # keep only rows with known Y
  colnames(trainDat)[1] <- 'Y'
  
  set.seed(1)
  registerDoParallel(cl)
  EN <- train(Y ~., data = trainDat, method = 'glmnet', trControl = fitControl, verbose = F) 
  ENlist[[k]] <- EN # save EN 
  ENimputed[is.na(tmpDat [ ,1]), imputeUS[k]] <- predict(EN, tmpDat[is.na(tmpDat [ ,1]), ])} # fill NAs with predictions 

names(ENlist) <- imputeUS


save(ENimputed, ENlist, adjDat, dogmzs, imputeUS, file='ENimputed')

rm(EN, ENlist)

#####################################################################
## how to evaluate?
#####################################################################
# try the 1/20ths scheme. cycle through data adding fake NAs to 1/20th of the data at a time, then compare imputed values to the held-out real data.

# divide the data by sample in to 1/20ths, which corresponds to missingess of 5%, and impute the data in each 20th and replace.

twentienths <- as.numeric(cut(1:nrow(mzdat), breaks=20)) 
set.seed(1)
random20ths <- sample(twentienths) # instead of cuting along the row order (which = runorder!!!), shuffle the 20ths. remember to set that seed
table(random20ths)
random20ths

ENevaluation.Randomized20ths <- mzdat # will replace with all imputed data for all mzs

fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 10)

##NOTE: the un-imputed input data will have mising values for Y, but not for the RF-imputed Ximp. This will give errors as it can't learn NA. For this test, use Ximp to get Y. 
for(k in 1:length(dogmzs)){
  Y <- select(Ximp, dogmzs[k])
  X <- select(Ximp, -dogmzs[k])
  tmpDat <- cbind(Y, X)
  
  for(z in 1:length(unique(random20ths))){
    trainDat <- tmpDat[random20ths!=z, ] # train model on rows without NA for the focal mz
    colnames(trainDat)[1] <- 'Y'
    set.seed(1)
    registerDoParallel(cl)
    EN <- train(Y ~., data = trainDat, method = 'glmnet', trControl = fitControl, verbose = F) 
    ENevaluation.Randomized20ths[random20ths==z, dogmzs[k]] <- predict(EN, tmpDat[random20ths==z, ]) # fill NAs with predictions 
  } }

save(ENevaluation.Randomized20ths, file='twentienths.testingENimputation')


load('ENimputed')
load('twentienths.testingENimputation')


melt(ENimputed)
melt(ENevaluation.Randomized20ths)

l <- melt(adjDat[ ,dogmzs]) # 'real' 
head(l)
colnames(l)[2] <- 'real'
l$RFimputed <- melt(Ximp)[ ,2]
l$Randomized.twentieths <- melt(ENevaluation.Randomized20ths)[ ,2]# 100% imputed

head(l)

colSums(is.na(l))

l$OrginallyMissing <- as.factor(ifelse(is.na(l$real), 'yes', 'no'))

ggplot(l, aes(y=real, x=Randomized.twentieths))+
  geom_point(size=0.1)+
  geom_abline(intercept=0, slope=1, color=2)+
  theme_classic(base_size = 7)+
  ggtitle('performance on known data')+
  facet_wrap(~variable, scales='free')


ggplot(subset(l, variable %in% dogmzs[c(1:10, 12:13)]), aes(y=real, x=Randomized.twentieths))+
  geom_point(size=0.1)+
  geom_abline(intercept=0, slope=1, color=2)+
  theme_classic(base_size = 16)+
  xlab('prediction')+
  ggtitle('performance on known (but held-out) data')+
  facet_wrap(~variable, scales='free')


# THIS is where you might compare to another method.  I tried knn.imutation, mice, and RFimpoutation, some were better than others, but EN imputation (above) was generally the most accurate at predicting held-out real data.

