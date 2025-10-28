# urine analysis

# Kate Creevy sent Daniel her criteria for diagnosing kidney disease using cbc chem and urine data:
# "If BUN and creatinine are high (above RI), when Usg is isosthenuric (1.008 - 1.012), then we diagnose kidney disease.
# If BUN and creatinine are high (above RI), when Usg is submaximal (<1.025), then we would strongly suspect kidney disease.  There are a variety of factors that may be at play here."

library(plyr)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(haven)


rm(list=ls())

setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')

u <- read.csv('data/2024_release/DAP_2024_SamplesResults_Urine_v1.0.csv')
head(u)
str(u)

## targeted LCMS data for P1 dogs merged with available cbc/chem data
load('data/scaled.data_for_CBC_mixedModel')
rm(grm)
b <- scDat

boxplot(b[ ,vars], pch=20, cex=0.5) # the variables in these data have been scaled

table(u$krt_urine_sg)
u <- u[u$krt_urine_sg<1.3, ] # remove the sample at urine sp > 1.3.  This was probably miscoded
u$krt_urine_sg[is.na(as.numeric(u$krt_urine_sg))] # for urine specific gravity, SOME values above 1.06 got symbolized ">1.060". For this analysis...
u$krt_urine_sg[is.na(as.numeric(u$krt_urine_sg))] <- 1.062 # code them as 1.062
u$krt_urine_sg <-  as.numeric(u$krt_urine_sg)
table(u$krt_urine_sg)

par(mfrow=c(2,2))
hist(as.numeric(u$krt_urine_sg), 30, border=0, col='grey40', las=1)

head(b)
table(b$cohort)
table(u$Sample_Year)


u$dog_id <- as.character(u$dog_id)
table(u$Sample_Year)
u <- u[u$Sample_Year == 'precision_1', ]
table(b$dog_id %in% u$dog_id)

colnames(u)

u <- merge(b, u, by='dog_id')


#####################################################################################################
# kidney disease?
# Kate Creevy sent Daniel her criteria for diagnosing kidney disease using cbc chem and urine data:
# "If BUN and creatinine are high (above RI), when Usg is isosthenuric (1.008 - 1.012), then we diagnose kidney disease.
# If BUN and creatinine are high (above RI), when Usg is submaximal (<1.025), then we would strongly suspect kidney disease.  There are a variety of factors that may be at play here."
#####################################################################################################
hist(u$krt_cp_bun_value)
u$krt_cp_bun_value <- exp(u$krt_cp_bun_value) # un-log bun values; a normalization had been done for modeling

# RI= Reference Interval = normal interval for patients

# [based on https]://www.vet.cornell.edu/animal-health-diagnostic-center/laboratories/clinical-pathology/reference-intervals/chemistry:
# RI creatinine = 0.6-1.4 mg/dL
# RI BUN = 9-26 mg/dL

# note from Kate:The RIs for TVMDL (where all DAP labs are performed) are
# BUN: 9-29 mg/dL
# creatinine: 0.56 - 1.40 mg/dL

upperRIcreatinine = 1.4
upperRIbun = 29
submaxSG = 1.025
isosthenuric <- c(1.008, 1.012)

table(u$krt_cp_bun_value > upperRIbun)
table(u$krt_cp_creatinine_value > upperRIcreatinine) # only 1 dog is above the RI for creatinine
table(u$krt_cp_bun_value > upperRIbun, u$krt_cp_creatinine_value > upperRIcreatinine) # that one dog also has above RI bun

sick_puppy <- u$krt_cp_bun_value > upperRIbun & u$krt_cp_creatinine_value > upperRIcreatinine
table(sick_puppy)

table(between(u$krt_urine_sg, isosthenuric[1], isosthenuric[2])) # 26 dogs within the isotheuric range
table(u$krt_urine_sg < submaxSG) # 130 dogs in the submaximal ureic sg range
table(between(u$krt_urine_sg, isosthenuric[1], isosthenuric[2]), sick_puppy) # the 'sick puppy is not isotheuric 
table(u$krt_urine_sg < submaxSG, sick_puppy)  # specific gravity of the is puppy urine is submaximal

par(mfrow=c(2,2))
plot(krt_urine_sg ~ sqrt(age), u, pch=20, col=rgb(0,0,0,0.5))
abline(h=isosthenuric[1], col=2, lwd=2)
abline(h=isosthenuric[2], col=3, lwd=2)

plot(krt_urine_sg ~ u$krt_cp_creatinine_value, u, pch=20, col=rgb(0,0,0,0.5))
abline(h=submaxSG, col=5)
abline(v=upperRIcreatinine, col=2)

table(u$krt_cp_bun_value > upperRIbun, u$krt_urine_sg <=1.012 & u$krt_urine_sg >=1.008)

plot(krt_urine_sg ~ u$krt_cp_bun_value, u, pch=20, col=rgb(0,0,0,0.5))
abline(h=isosthenuric[1], col=2, lwd=2)
abline(h=isosthenuric[2], col=3, lwd=2)
abline(v=upperRIbun, col=4)
abline(h=submaxSG, col=5)

# as of April 2024, one P1 dog meets the criteria of "strongly suspect kidney disease" described above.  see: sick puppy


########################################################
## covariates from the urinalysis:
########################################################
uvars <- colnames(u)[grepl('krt_urine', colnames(u))]

str(u[ ,uvars])

apply(u[ ,uvars], 2, table)
table(u[ ,uvars[1]])
table(u[ ,uvars[2]])
table(u[ ,uvars[3]])
table(u[ ,uvars[5]])

apply(u[ ,uvars], 2, function(x) var(x, na.rm=T)) # measure varance among UA traits
apply(u[ ,uvars], 2, function(x) var(as.numeric(x, na.rm=T)))
n <- apply(u[ ,uvars], 2, function(x) is.na(x))
colSums(n) # nas per variable
uvars <- uvars[colSums(n) <10] # remove variables with >10 missing values, goes from 31 to 18

urine.tech.vars <- c('krt_urine_col_meth', 'krt_urine_volume') # these urine variables could be considered 'technical' (or 'technique'). add these to models as covars

## some variables are invariant, remove them:
apply(u[ ,uvars[1:9]], 2, table)
apply(u[ ,uvars[10:18]], 2, table)

invariants <- c('krt_urine_glucose', 'krt_urine_ketones') # invariants
uvars <- uvars[!uvars %in% invariants] # 16 remain 

#####################################################
## some variables are factors, order their levels:
#####################################################.  GO SLOW, you can easily drop observations doing this
apply(u[ ,uvars], 2, table)

sum(print(table(u$krt_urine_protien)))
u$krt_urine_protien <- factor(u$krt_urine_protien, levels=c('Negative', 'Trace', '1+', '2+', '3+'))
sum(print(table(u$krt_urine_protien)))

sum(print(table(u$krt_urine_blood_hbg)))
u$krt_urine_blood_hbg <- factor(u$krt_urine_blood_hbg, levels=c('Negative', 'Trace', '1+', '2+', '3+'))
sum(print(table(u$krt_urine_blood_hbg)))

sum(print(table(u$krt_urine_bilirubin)))
u$krt_urine_bilirubin <- factor(u$krt_urine_bilirubin, levels=c('Negative', '1+', '2+'))
sum(print(table(u$krt_urine_bilirubin)))

sum(print(table(u$krt_urine_fat_hpf)))
u$krt_urine_fat_hpf <- factor(u$krt_urine_fat_hpf, levels=c('None Observed', 'Rare', 'Few', 'Moderate', 'Many'))
sum(print(table(u$krt_urine_fat_hpf)))

sum(print(table(u$krt_urine_urothelial_hpf)))
u$krt_urine_urothelial_hpf <- factor(u$krt_urine_urothelial_hpf, levels=c('None Observed', 'Rare', 'orig Few harm 0-3', 'orig 0-5 harm 0-3', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count')) # I don't know what orig/harm means (sent email to Kate Creevy)
sum(print(table(u$krt_urine_urothelial_hpf)))

sum(print(table(u$krt_urine_squamous_hpf)))
u$krt_urine_squamous_hpf <- factor(u$krt_urine_squamous_hpf, levels=c('None Observed', 'Rare', 'orig Few harm 0-3', 'orig 0-5 harm 0-3', 'orig 5-10 harm 6-10', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count'))
sum(print(table(u$krt_urine_squamous_hpf)))

sum(print(table(u$krt_urine_rbc_hpf)))
u$krt_urine_rbc_hpf <- factor(u$krt_urine_rbc_hpf, levels=c('None Observed', 'Rare', 'orig Few harm 0-3', 'orig 0-5 harm 0-3', 'orig 5-10 harm 6-10', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count'))
sum(print(table(u$krt_urine_rbc_hpf)))

sum(print(table(u$krt_urine_wbc_hpf)))
u$krt_urine_wbc_hpf <- factor(u$krt_urine_wbc_hpf, levels=c('None Observed', 'Rare', 'orig Few harm 0-3', 'orig 0-5 harm 0-3', 'orig 5-10 harm 6-10', 'orig 10-20 harm 10-10','orig 20-40 harm 20-40', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count'))
sum(print(table(u$krt_urine_wbc_hpf)))


##################################################################################################################################
# 'harm' stands for 'harmonized', which was a correction done to make different labs/vets/scoring systems harmoneous.
##################################################################################################################################
## so, recode:

sum(print(table(u$krt_urine_urothelial_hpf)))
u$krt_urine_urothelial_hpf[u$krt_urine_urothelial_hpf=='orig Few harm 0-3'] <- '0-3'
u$krt_urine_urothelial_hpf[u$krt_urine_urothelial_hpf=='orig 0-5 harm 0-3'] <- '0-3'
u$krt_urine_urothelial_hpf <- factor(u$krt_urine_urothelial_hpf, levels=c('None Observed', 'Rare', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count')) # I don't know what orig/harm means (sent email to Kate Creevy)
sum(print(table(u$krt_urine_urothelial_hpf)))

sum(print(table(u$krt_urine_squamous_hpf)))
u$krt_urine_squamous_hpf[u$krt_urine_squamous_hpf== 'orig Few harm 0-3'] <- '0-3'
u$krt_urine_squamous_hpf[u$krt_urine_squamous_hpf== 'orig 0-5 harm 0-3'] <- '0-3'
u$krt_urine_squamous_hpf[u$krt_urine_squamous_hpf== 'orig 5-10 harm 6-10'] <- '6-10'
u$krt_urine_squamous_hpf <- factor(u$krt_urine_squamous_hpf, levels=c('None Observed', 'Rare', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count'))
sum(print(table(u$krt_urine_squamous_hpf)))

sum(print(table(u$krt_urine_rbc_hpf)))
u$krt_urine_rbc_hpf[u$krt_urine_rbc_hpf== 'orig Few harm 0-3'] <- '0-3'
u$krt_urine_rbc_hpf[u$krt_urine_rbc_hpf== 'orig 0-5 harm 0-3'] <- '0-3'
u$krt_urine_rbc_hpf[u$krt_urine_rbc_hpf== 'orig 5-10 harm 6-10'] <- '6-10'
u$krt_urine_rbc_hpf <- factor(u$krt_urine_rbc_hpf, levels=c('None Observed', 'Rare', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count'))
sum(print(table(u$krt_urine_rbc_hpf)))

sum(print(table(u$krt_urine_wbc_hpf)))
u$krt_urine_wbc_hpf[u$krt_urine_wbc_hpf== 'orig Few harm 0-3'] <- '0-3'
u$krt_urine_wbc_hpf[u$krt_urine_wbc_hpf== 'orig 0-5 harm 0-3'] <- '0-3'
u$krt_urine_wbc_hpf[u$krt_urine_wbc_hpf== 'orig 5-10 harm 6-10'] <- '6-10'
u$krt_urine_wbc_hpf[u$krt_urine_wbc_hpf== 'orig 10-20 harm 10-10'] <- '10-20'
u$krt_urine_wbc_hpf[u$krt_urine_wbc_hpf== 'orig 20-40 harm 20-40'] <- '20-40'
u$krt_urine_wbc_hpf <- factor(u$krt_urine_wbc_hpf, levels=c('None Observed', 'Rare', '0-3', '3-6', '6-10', '10-20', '20-40', 'Too Numerous to Count'))
sum(print(table(u$krt_urine_wbc_hpf)))

#################################################################
## I don't know how to order levels (or if order is appropriate) for: urine color, urine transparency, urine crystals, urine casts
u$krt_urine_color <- as.factor(u$krt_urine_color)
u$krt_urine_transparency <- as.factor(u$krt_urine_transparency)
u$krt_urine_crystals <- as.factor(u$krt_urine_crystals)
u$krt_urine_casts <- as.factor(u$krt_urine_casts)

str(u[ ,uvars])

save(u, urine.tech.vars, uvars, dogmzs, cellCovars, vars, file='dog metabolome paper/dataForUrinalysis')
#################################################################


rm(list=ls())
#########################################################################################################
# test age associations of urine variables in full model with covariates + grm
#########################################################################################################

load('data/scaled.data_for_CBC_mixedModel')
load('dog metabolome paper/dataForUrinalysis')

str(u [, uvars])

uvars <- uvars[!uvars %in% urine.tech.vars] # remove technical covaraites from uvars

method <- as.factor(u[ ,urine.tech.vars[1]])
volume <- as.factor(u[ ,urine.tech.vars[2]])

table(method) # consolidate method
method[method %in% c('Unknown', 'Voided')] <- 'Free Catch'
method <- droplevels(method)

table(volume) # order volume
volume <- factor(volume, levels=c('<1 mL', '1-3 mL', '>3 mL'))
table(volume) 

u$method <- method
u$volume <- volume

colSums(is.na(u[ ,uvars])) # remaining missingness
table(colSums(is.na(u[ ,uvars])))
table(rowSums(is.na(u[ ,uvars]))) 
u <- u[rowSums(is.na(u[ ,uvars]))==0, ] # keep only complete samples, takes n=714 to n=738 dogs

str(u [, uvars])
uvars <- uvars[!uvars %in% urine.tech.vars] # remove technical covariates from uvars

method <- as.factor(u[ ,urine.tech.vars[1]])
volume <- as.factor(u[ ,urine.tech.vars[2]])

table(method) # consolidate method
method[method %in% c('Unknown', 'Voided')] <- 'Free Catch'
u$method <- droplevels(method)

table(volume) # there is an issue with the minority '<1 mL' vol sample, (not in some boots), combine to overcome
volume[volume=='<1 mL'] = '1-3 mL'
u$volume <- as.factor(volume)

table(u$dog_id %in% rownames(grm))
grm <- grm[u$dog_id, u$dog_id] # subset and order the GRM by the dogs in the mz data

boxplot(u[ ,vars]) # check to see that the data have been scaled
Zmat <- diag(nrow(u)) # an empty matrix for the random effects

mzdat <- u[ ,dogmzs]


# convert (logically ordered) factors to numeric, so that they become single variables
str(u[ ,uvars])

u$krt_urine_protien <- as.numeric(u$krt_urine_protien) 
u$krt_urine_bilirubin <- as.numeric(u$krt_urine_bilirubin) 
u$krt_urine_blood_hbg <- as.numeric(u$krt_urine_blood_hbg) 
u$krt_urine_wbc_hpf <- as.numeric(u$krt_urine_wbc_hpf) 
u$krt_urine_rbc_hpf <- as.numeric(u$krt_urine_rbc_hpf) 
u$krt_urine_squamous_hpf <- as.numeric(u$krt_urine_squamous_hpf) 
u$krt_urine_fat_hpf <- as.numeric(u$krt_urine_fat_hpf) 
u$krt_urine_urothelial_hpf <- as.numeric(u$krt_urine_urothelial_hpf) 

uvars
testUvars <- uvars[c(2,3,4,5,6,7,8,9,10,13)] # select numerics and ordered factors

Udat <- scale(u[ ,testUvars])
boxplot(Udat)


save(Udat, u, testUvars, vars, Zmat, grm, dogmzs, cellCovars, file='dog metabolome paper/UrineMixedModelData')

rm(list=ls())
######################################################################
load('dog metabolome paper/UrineMixedModelData')

tmp <- ' ~ sqrtAge + sqrtWT + sex * sterilization_status + hours_fasting'
tmp2 <- ', data=u)'
cat(c(tmp, paste0('+ ', cellCovars)), tmp2)

X <- model.matrix(   ~ sqrtAge + sqrtWT + sex * sterilization_status + hours_fasting + krt_cbc_total_wbcs + krt_cbc_abs_bands + krt_cbc_abs_neutrophils + krt_cbc_abs_lymphocytes + krt_cbc_abs_monocytes + krt_cbc_abs_eosinophils + krt_cbc_abs_basophils + krt_cbc_rbc + krt_cbc_hgb + krt_cbc_hct + krt_cbc_mcv + krt_cbc_mch + krt_cbc_mchc + krt_cbc_rdw + krt_cbc_mpv + krt_cbc_pct + krt_cbc_retic_abs , data=u)

###################################
# the next steps can be slow, use parallel processing
library(parallel)
library(doParallel)
library(EMMREML)

n.cores <- detectCores(all.tests = F, logical = T) # detects the number of available cores
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  

emma <- emmreml(y=Udat[ ,1], X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) # the model fitting step, run an example to see what it does and to record the output rownames etc. (the code below will call the output)
head(emma)

# get fixed effects
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  
clusterExport(clus, varlist=c("Udat", 'X', 'Zmat', 'grm'), envir = environment()) 

fixedEffects <- t(parApply(clus, Udat, 2, function(Y) {
  library(EMMREML)
  emma <- emmreml(y=Y, X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) 
  p=emma$pvalbeta
  b=emma$betahat
  return(c(b, p[ ,"none"])) } ))

fixedEffects <- as.data.frame(fixedEffects)

betas <- fixedEffects[ ,1:length(rownames(emma$betaha))]
colnames(betas) <- rownames(emma$betahat)  
Ps <- fixedEffects[ ,c((1+length(rownames(emma$betaha))):ncol(fixedEffects))]
colnames(Ps) <- rownames(emma$betahat)  

fdr <- as.data.frame(apply(Ps, 2, function(x) p.adjust(x, 'fdr')))
colSums(fdr<=0.05)


testUvars[fdr$sqrtAge<=0.05] # two UA traits assoc with age

u$krt_urine_bilirubin <- as.factor(u$krt_urine_bilirubin)
levels(u$krt_urine_bilirubin) <- c('Negative', '1+', '2+') # convert bilirubin back to factor for plotting


ggplot(u, aes(y=krt_urine_sg, x=sqrt(AgeAtDOC)))+
  geom_smooth(method=lm, size=0)+
  geom_point()+
  ylab('urine specific gravity')+
  xlab(bquote(age~(sqrt(years))))+
  theme_classic(base_size = 14)

ggplot(subset(u, !is.na(krt_urine_bilirubin)), aes(y=krt_urine_bilirubin, x=sqrt(AgeAtDOC)))+
  geom_jitter(height=0.1, alpha=0.5)+
  ylab('urine bilirubin')+
  xlab(expression("age" ~ sqrt(years)))+
  theme_classic(base_size = 14)

ggarrange(p1, p2)



####@####################################################################################
# Question from Katie Tolburg
# any metabolites assoc with urine protein?
#########################################################################################

# any metabolites assoc with ANY urine variables?
rm(list=ls())
######################################################################
load('dog metabolome paper/UrineMixedModelData')

str(u[ ,testUvars])
scu <- u
scu[ ,testUvars] <- scale(scu[ ,testUvars]) # scale the varaibles

par(mfrow=c(1,2))
boxplot(u[ ,c(testUvars, cellCovars, dogmzs)], cex=0.3, pch=19, col='grey', 'unscaled vars')
points(colMeans(u[ ,c(testUvars, cellCovars, dogmzs)]), col=2, pch=19, cex=0.5)
boxplot(scu[ ,c(testUvars, cellCovars, dogmzs)], cex=0.3, pch=19, col='grey', 'scaled vars')
points(colMeans(scu[ ,c(testUvars, cellCovars, dogmzs)]), col=2, pch=19, cex=0.5)

mzdat <- scu[ ,dogmzs]

tmp <- ' ~ sqrtAge + sqrtWT + sex * sterilization_status + hours_fasting'
tmp2 <- ', data=u)'

cat(c(tmp, paste0('+ ', testUvars), paste0('+ ', cellCovars)), tmp2)

X <- model.matrix(  ~ sqrtAge + sqrtWT + sex * sterilization_status + hours_fasting + krt_urine_sg + krt_urine_bilirubin + krt_urine_blood_hbg + krt_urine_ph + krt_urine_protien + krt_urine_wbc_hpf + krt_urine_rbc_hpf + krt_urine_squamous_hpf + krt_urine_urothelial_hpf + krt_urine_fat_hpf + krt_cbc_total_wbcs + krt_cbc_abs_bands + krt_cbc_abs_neutrophils + krt_cbc_abs_lymphocytes + krt_cbc_abs_monocytes + krt_cbc_abs_eosinophils + krt_cbc_abs_basophils + krt_cbc_rbc + krt_cbc_hgb + krt_cbc_hct + krt_cbc_mcv + krt_cbc_mch + krt_cbc_mchc + krt_cbc_rdw + krt_cbc_mpv + krt_cbc_pct + krt_cbc_retic_abs , data=scu)

###################################
# the next steps can be slow, use parallel processing
library(parallel)
library(doParallel)
library(EMMREML)

n.cores <- detectCores(all.tests = F, logical = T) # detects the number of available cores
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  

emma <- emmreml(y=scu$`Maleic Acid`, X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) # the model fitting step, run an example to see what it does and to record the output rownames etc. (the code below will call the output)
head(emma)

clusterExport(clus, varlist=c("mzdat", 'dogmzs', 'X', 'Zmat', 'grm'), envir = environment()) # specifies what things in the R environment that will be accessible to parallel processing function

# get fixed effects
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  
clusterExport(clus, varlist=c("mzdat", 'dogmzs', 'X', 'Zmat', 'grm'), envir = environment()) 

fixedEffects <- t(parApply(clus, mzdat, 2, function(Y) {
  library(EMMREML)
  emma <- emmreml(y=Y, X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) 
  p=emma$pvalbeta
  b=emma$betahat
  return(c(b, p[ ,"none"])) } ))

fixedEffects <- as.data.frame(fixedEffects)

# get random effects (BLUPs)
randomEffects <- parApply(clus, mzdat, 2, function(Y) {
  library(EMMREML)
  emma <- emmreml(y=Y, X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) 
  blup <- emma$uhat
  varblup <- emma$varuhat
  return(list(blup)) } )

colnames(fixedEffects) <- c(paste0('beta_', rownames(emma$betahat)), paste0('P_', rownames(emma$betahat)))
betas <- fixedEffects[ ,1:length(rownames(emma$betaha))]
colnames(betas) <- rownames(emma$betahat)  
Ps <- fixedEffects[ ,c((1+length(rownames(emma$betaha))):ncol(fixedEffects))]
colnames(Ps) <- rownames(emma$betahat)  

fdr <- as.data.frame(apply(Ps, 2, function(x) p.adjust(x, 'fdr')))
colSums(fdr<=0.05) # 15 mzs assoc with uSG.
# also, 'when controlling for the other urine variables', there is one mz assoc, with urine protein (contrary to the prev. section)

save(u, scu, fdr, betas, fixedEffects, randomEffects, X, dogmzs, cellCovars, file='dog metabolome paper/UrineMixedModelResults')


dogmzs[fdr$krt_urine_protien<=0.05] # ornitine assoc. with urine protein
dogmzs[fdr$krt_urine_sg<=0.05] # of the ptmAAs, hydroxyproline is assoc. with uSG
dogmzs[fdr$krt_urine_fat_hpf<=0.05]
dogmzs[fdr$krt_urine_blood_hbg<=0.05]


u$krt_urine_protien <- factor(u$krt_urine_protien, levels=c('Negative', 'Trace', '1+', '2+', '3+'))

dev.off()
ggplot(scu, aes(y=Ornithine, x=krt_urine_protien))+
  geom_jitter(width=0.05)+
  theme_bw(base_size = 14)+
  geom_smooth(method='lm')


rm(list=ls())
##################################################################################
load('dog metabolome paper/mediationByCREAT')
load('dog metabolome paper/UrineMixedModelResults')

colSums(fdr<=0.05)


u$krt_urine_protien <- as.factor(c('Negative', 'Trace', '1+', '2+', '3+')[u$krt_urine_protien])
u$krt_urine_protien <- factor(u$krt_urine_protien, levels= c('Negative', 'Trace', '1+', '2+', '3+'))
u$krt_urine_protien

ggplot(u, aes(y=Ornithine, x=as.numeric(krt_urine_protien)))+
  geom_jitter(width=0.05)+
  theme_bw(base_size = 14)+
  geom_smooth(method='lm')
# this relationship may be under-represented graphically due to effects of covariates



x <- fdr$krt_urine_sg[order(fdr$krt_urine_sg)]
names(x) <- dogmzs[order(fdr$krt_urine_sg)]
uSGmzs <- names(x[x<=0.05]) # 15 mzs assoc. with urine specific gravity


usgVolc <- data.frame(mz=rownames(betas), uSGbeta = betas$krt_urine_sg, P = fixedEffects$P_krt_urine_sg, FDR = fdr$krt_urine_sg)


volc <- ggplot(usgVolc, aes(y=-log(P, 10), x=uSGbeta, label=mz, color=FDR<=0.05))+
  geom_point( )+
  theme_classic(base_size = 16)+
  ggrepel::geom_text_repel(max.overlaps = 6)+
  labs(y=expression(-log[10]~P))+
  xlab(expression(beta ~ uSG))+
  theme(legend.position = 'none')+
  scale_color_manual(values=c('grey40', 'red'))

volc

sub <- subset(usgVolc, mz %in% uSGmzs)
sub <- sub[order(sub$uSGbeta), ]
sub$mz <- factor(sub$mz, levels=sub$mz)
sub$mz

betaP <- ggplot(sub, aes(x=mz, y=uSGbeta)) + 
  geom_bar(stat = "identity")+
  coord_flip()+
  theme_classic(base_size = 14)+
  xlab('')+
  ylab(expression(beta ~ uSG))

betaP 


