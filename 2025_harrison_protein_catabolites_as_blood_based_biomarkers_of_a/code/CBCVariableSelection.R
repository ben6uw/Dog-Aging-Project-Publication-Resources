# blood chem covariates:
library(plyr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(haven)
library(ordinal)

rm(list=ls())

getwd()
setwd("G:/My Drive/Documents/DogAgingProject")
setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')

# there are several types of data about the blood samples, 'CBC' (complete blood count, counts of blood cell types etc.), and 'ChemistryPanel' blood chemistry measurements.   
dir('data/2024_release')


cbc <- read.csv("data/2024_release/DAP_2024_SamplesResults_CBC_v1.0.csv") # cbc data as of 2024 release
colnames(cbc)[colnames(cbc)=='Sample_Year'] <- 'cohort'
cbc$cohort <- gsub('precision_', 'Precision ', cbc$cohort)
head(cbc)
cbc <- cbc[ ,!grepl('comments', colnames(cbc))]# this step omits the "krt_cbc_test_comments"
cbc <- cbc[ ,!grepl('DAP_Sample_ID', colnames(cbc))]


chem <- read.csv('data/2024_release/DAP_2024_SamplesResults_ChemistryPanel_v1.0.csv', stringsAsFactors = T) # cbc data as of 2024 release
head(chem)
colnames(chem)[colnames(chem)=='Sample_Year'] <- 'cohort'
chem$cohort <- gsub('precision_', 'Precision ', chem$cohort)
head(chem)
chem<- chem[ ,!grepl('comments', colnames(chem))] # this step omits the comments
chem<- chem[ ,!grepl('DAP_Sample_ID', colnames(chem))] 

b <- merge(cbc, chem, by=c('dog_id', 'cohort')) # make b (blood)

bloodvars <- colnames(b)[-c(1:3)]
bloodvars

table(grepl('cbc', bloodvars)) # 45 cbc cell traits, and 42 chem traits


b <- b[b$cohort=="Precision 1", ]

# to simplify, remove invariant bloodvars
invars <- apply(b[ ,bloodvars], 2, function(x) var(as.numeric(x), na.rm=T))
invars <- invars[!is.na(invars)] 
table(invars==0) # 4 invariants
invars <- names(invars)[invars==0] 
bloodvars <- bloodvars[!bloodvars %in% invars] # remove 4 invariants

str(b[ ,bloodvars])
bloodvars <- bloodvars[bloodvars!='krt_cbc_hemoparasites'] # another invariant
colSums(is.na(b[ ,bloodvars])) # some of these could drop out due to NAs

miss <- colSums(is.na(b[ ,bloodvars]))
table(miss)
mean(miss)

# any covariate with NA won't enable 'adjustment' for its effect, unless the NAs are imputed
table(colSums(is.na(b[ ,bloodvars])) < 10)
# no chance to imput NAs when too common, chop by number of NAS:
bloodvars <- bloodvars[colSums(is.na(b[ ,bloodvars])) < 10] # this step leaves 46 blood covars
table(grepl('cbc', bloodvars))

## remove all 'rel' measures (I worked with these and they are all redundant to the abs variables, and, due to dependence on other measures, are somewhat ambiguous)
bloodvars <- bloodvars[!grepl('rel', bloodvars)]
bloodvars <- bloodvars[!grepl('_per', bloodvars)]

colSums(is.na(b[ ,bloodvars])) # for the continuous vars, transform, then impute
str(b[ ,bloodvars])

# consider removing zero-inflated covars
par(mfrow=c(2,2))
plot(b$krt_cbc_abs_bands) 
plot(b$krt_cbc_abs_basophils)
plot(b$krt_cbc_rel_bands) # these are discrete (counts), in which zero inflation is fine
plot(b$krt_cbc_rel_basophils) # these are discrete (counts), in which zero inflation is fine

str(b[ ,bloodvars])

table(grepl('cbc', bloodvars))

# before imputation:
# take the numerics
bloodnums <- dplyr::select_if(b[ ,bloodvars], is.numeric)

norms <- apply(bloodnums, 2, shapiro.test) # test for normal dist
stat <- sapply(norms, function(x) x$statistic)
pval <- sapply(norms, function(x) x$p.value)
dim(bloodnums)

par(mfrow=c(4,6))
for(i in 1:ncol(bloodnums)){
  hist(bloodnums[ ,i], main=round(stat[i], 3), xlab=colnames(bloodnums)[i], ylab='') }

table('normal'=stat<0.96, ifelse(grepl('cbc', colnames(bloodnums)), 'cell', 'chem')) # a shapiro test stat of <0.96 seems a good threshold for log transformation

tmp <- data.frame(colnames(bloodnums), stat)
colnames(tmp)[1] <- c('var')

gsub('krt_', '', tmp$var[tmp$stat<0.96 & grepl('cbc', tmp$var)])

loggedBloodvars <- colnames(bloodnums)[stat<0.96]
loggedBloodvars <- gsub('krt_cbc_', '', loggedBloodvars)
loggedBloodvars <-  gsub('krt_cp_', '', loggedBloodvars)
loggedBloodvars <-  gsub('_value', '', loggedBloodvars)
loggedBloodvars

bloodnums[ ,stat<0.96] <- log(bloodnums[ ,stat<0.96]+1)
str(b[ ,bloodvars])

# after log-normalization
norms <- apply(bloodnums, 2, shapiro.test) # test for normal dist
stat <- sapply(norms, function(x) x$statistic)
pval <- sapply(norms, function(x) x$p.value)

par(mfrow=c(4,4))
for(i in 1:ncol(bloodnums)){
  hist(bloodnums[ ,i], main=round(stat[i], 3), xlab=colnames(bloodnums)[i]) }
# check for stragglers

# straggler
par(mfrow=c(2,2))
hist(b$krt_cp_ggt_value)
hist(log(b$krt_cp_ggt_value))
table(b$krt_cp_ggt_value)
table(log(b$krt_cp_ggt_value)) # ggt has a tail, but is digital

colSums(is.na(bloodnums)) # now for imputation
miss <- colSums(is.na(bloodnums))
mean(miss) # mean missingness per variable
max(miss) # maximum missingness per variable
mean(miss[miss>0])

imp <- t(bloodnums)
imp <- impute::impute.knn(imp)
imp <- t(imp$data)
b[ ,colnames(imp)] <- imp # replace with log-transformed and imputed data

colSums(is.na(b))
b <- b[ ,colSums(is.na(b)) == 0] # remove the last vars with NAs, these should just be non-numerics
str(b)

bloodvars <- colnames(b)[-c(1,2)] # 38 bloodvars

table(grepl('cbc', bloodvars)) # 17 cell traits, 21 chem traits

save(b, bloodvars, loggedBloodvars, file='dog metabolome paper/data/blood.covariates.RData')


rm(list=ls())
###########################################################
# remove techcnical effects; these blood samples, like those for metabolomics, were 'Shipped to Texas' and so transit time and arrival temperature should be considered
load('dog metabolome paper/data/blood.covariates.RData')
samp <- read.csv('dog metabolome paper/data/P1sampleinfo.csv')

samp[1:4, 1:4]
samp$dog_id <- as.character(samp$dog_id)

str(samp)
hist(samp$elapse_DOC_LOG) # transit time
hist(samp$serum_temp) # arrival temperature

b <- merge(samp[ ,c('dog_id', 'elapse_DOC_LOG', 'serum_temp')], b)

median(b$serum_temp)
range(b$serum_temp)

pList <- list()

for(i in 1:length(bloodvars)){
s <- summary(lm(b[ ,bloodvars[i]] ~ elapse_DOC_LOG + serum_temp, b))
pList[[i]] <- s$coefficients[2:3, 4]}

names(pList) <- bloodvars
p <- as.data.frame(do.call(rbind, pList))

p[order(p$elapse_DOC_LOG), ][1:6, ]

ggplot(b, aes(y=krt_cbc_mchc, x=elapse_DOC_LOG))+
  geom_point()+
  theme_bw(base_size=18)+
  geom_smooth(method='lm', size=0)

ggplot(b, aes(y=krt_cbc_mcv, x=elapse_DOC_LOG))+
  geom_point()+
  theme_bw(base_size=18)


mean(b$elapse_DOC_LOG)
median(b$elapse_DOC_LOG)
range(b$elapse_DOC_LOG)

p[order(p$serum_temp), ][1:6, ]

ggplot(b, aes(y=krt_cbc_mpv, x=elapse_DOC_LOG))+
  geom_point(alpha=0.5)+
  theme_bw(base_size=18)

ggplot(b, aes(y=krt_cp_magnesium_value, x=elapse_DOC_LOG))+
  geom_point(alpha=0.5)+
  theme_bw(base_size=18)

ggplot(b, aes(y=krt_cbc_rdw, x=elapse_DOC_LOG))+
  geom_point(alpha=0.5)+
  theme_bw(base_size=18)+
  geom_smooth(method='lm', size=0)

## linear correction seems appropriate (ie, none are particularly non-linear)
# adjust for travel time and arrival temp
for(i in 1:length(bloodvars)){
  b[ ,bloodvars[i]] <- residuals(lm(b[ ,bloodvars[i]] ~ elapse_DOC_LOG + serum_temp, b))}

save(b, bloodvars, file='dog metabolome paper/data/blood.covariates.RData')










