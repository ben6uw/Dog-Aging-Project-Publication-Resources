library(plyr)
library(dplyr)
library(tidyverse)
library(pROC)
library(gridExtra)
library(ggpubr)
library(ggplot2)
library(mediation)
library(parallel)
library(doParallel)

setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')

rm(list=ls())
###################################
load('data/scaled.data_for_CBC_mixedModel')


###################################
# mediation analysis: creatinine:
###################################

sensitiveTest <- function(x) ifelse(x$ind.d0[x$rho==0]==0, 'passes', 'fails')
medOut<- function(x) data.frame('Total Effect'= x$tau.coef, 'ADE' = x$z0, 'ADElowerCI' = x$z0.ci[1], 'ADEupperCI'= x$z0.ci[2], 'ADE_P' = x$z0.p, 'ACME' = x$d0, 'ACMElowerCI' = x$d0.ci[1], 'ACMEupperCI'= x$d0.ci[2], 'ACME_P' = x$d0.p, 'ADEpropOfTotal Effect'=x$z0/x$tau.coef)

medModList <- list()
clusterExport(clus, varlist=c('scDat', 'dogmzs'), envir = environment()) 

for (i in 1:length(dogmzs)) {
  scDat$MZ <- scDat[ ,dogmzs[i]]
  med.fit <- lm(MZ ~ sqrtAge + sqrtWT + hours_fasting + sex * sterilization_status, data = scDat) 
  out.fit <- lm(MZ ~ krt_cp_creatinine_value + sqrtAge + sqrtWT + hours_fasting + sex * sterilization_status, data = scDat)
  medModList[[i]] <- mediate(med.fit, out.fit, treat = "sqrtAge", mediator = "krt_cp_creatinine_value",  sims=10000, boot=T) 
}

names(medModList) <- dogmzs
sensList <- lapply(medModList, function(x) medsens(x))
sumList <- lapply(medModList, function(x) summary(x))
out <- lapply(sumList, medOut)
out <- do.call(rbind, out)
out$mz <- dogmzs
out$ADE_FDR <- p.adjust(out$ADE_P, 'fdr')
out$ACME_FDR <- p.adjust(out$ACME_P, 'fdr')
out$sensitivity <- sapply(sensList, sensitiveTest)
out$mediator <- 'krt_cp_creatinine_value'
save(out, medModList, sensList, file='dog metabolome paper/mediationByCREAT')




rm(list=ls())
################
load('dog metabolome paper/mediationByCREAT')
out[out$ACME_FDR<=0.05, ]

plot(medModList[['N-Ac-Phenylalanine']])

# how often are the ptmAAs mediated by creat?
mods <- read.csv('dog metabolome paper/modAAs.csv')
table(mods$type)
out$PTM <- out$mz %in% mods$mz[mods$type=='PTM']

table('mediated'=out$ACME_FDR<=0.05, 'ptmAA'=out$PTM) # but, how many of these ptmAAs are age-associated????
fisher.test(table('mediated'=out$ACME_FDR<=0.05, 'ptmAA'=out$PTM))

a <- read.table('dog metabolome paper/TableS1.txt', sep='\t', head=T)
head(a)
colnames(a)[colnames(a)=='FDR'] <- 'age_FDR'
colnames(a)[colnames(a)=='age.Beta'] <- 'B_age'

head(a)

x <- bind_cols(out[a$Metabolite, ], a[ ,c('Metabolite', 'B_age', 'P.value', 'age_FDR')])


table('mediated'=x$ACME_FDR<=0.05, 'ptmAA'=x$PTM, 'age.assoc'=x$age_FDR<=0.05) # the age effect of 4 of the 7 age-assoc ptmAAs are mediated by creatinine. whereas none of the 6 non-age-assoc mzs are medaited 
ptmAAsMediated <- print(x$mz[x$ACME_FDR<=0.05 & x$PTM & x$age_FDR<=0.05]) # age-assoc. ptmAAs mediated by creat
x$mz[x$ACME_FDR<=0.05 & x$PTM & x$age_FDR<=0.05 & x$B_age>0]
x$mz[x$ACME_FDR<=0.05 & x$PTM & x$age_FDR<=0.05 & x$B_age<0]

barplot(out$ADEpropOfTotal.Effect[out$mz %in% ptmAAsMediated])
hist(out$ADEpropOfTotal.Effect[out$ACME_FDR<=0.05])


sumList <- lapply(medModList, function(x) summary(x))
p <- lapply(sumList, function(x) data.frame('propMed'=x$n0, 'LCI'=x$n.avg.ci[1], 'UCI'=x$n.avg.ci[2]))
p <- do.call(rbind, p)
p$mz <- rownames(p)

head(p)
head(out)

p <- left_join(p, dplyr::select(out, c(mz, ACME_FDR)))

mediationCandidates <- c("Dimethylarginine (A/SDMA)",  "Hydroxyproline",  "N6-Acetyl-Lysine",  "N6-Trimethyllysine",  "N-Ac-Alanine",  "N-Ac-L-Glutamine",  "N-Ac-Phenylalanine",  "N-Ac-Tryptophan",  "N-Ac-Aspartate",  "N-AcetylGlycine",  "n-Formylmethionine")

head(p)


p <- p[p$mz %in% x$mz[x$age_FDR<=0.05], ] # limit to only those age-assoc. ptmAAs

p$sig <- as.factor(ifelse(p$ACME_FDR>0.05, 'NS', 'sig'))

p$mz[p$mz=="N-Acetyl-Aspartate (NAA)"] <- "N-Ac-Aspartate"
p$mz <- factor(p$mz, levels=p$mz[order(p$propMed)])

creatMedPlot <- ggplot(subset(p, mz %in% mediationCandidates), aes(y=propMed, x=mz, fill=sig))+
  geom_bar(stat='identity', color='black') + 
  geom_errorbar(aes(ymax = UCI, ymin=LCI, width=0.1))+
  theme_bw(base_size = 16)+
  theme(legend.position = 'none')+
  scale_fill_manual(values=c('NS'='white', 'sig'='grey40'))+
  coord_flip(ylim = c(0, 1)) + 
  xlab('')+
  ylab('proportion mediated (+/- 95%CI)')+
  ggtitle('mediation by serum creatinine')

creatMedPlot 

save(creatMedPlot, file='dog metabolome paper/plots/creatMedPlot')




###################################
# mediation analysis: BUN:
###################################

rm(list=ls())
###################################
load('data/scaled.data_for_CBC_mixedModel')

sensitiveTest <- function(x) ifelse(x$ind.d0[x$rho==0]==0, 'passes', 'fails')
medOut<- function(x) data.frame('Total Effect'= x$tau.coef, 'ADE' = x$z0, 'ADElowerCI' = x$z0.ci[1], 'ADEupperCI'= x$z0.ci[2], 'ADE_P' = x$z0.p, 'ACME' = x$d0, 'ACMElowerCI' = x$d0.ci[1], 'ACMEupperCI'= x$d0.ci[2], 'ACME_P' = x$d0.p, 'ADEpropOfTotal Effect'=x$z0/x$tau.coef)

medModList <- list()


n.cores <- detectCores()
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  
clusterExport(clus, varlist=c('scDat', 'dogmzs'), envir = environment()) 


for (i in 1:length(dogmzs)) {
  scDat$MZ <- scDat[ ,dogmzs[i]]
  med.fit <- lm(MZ ~ sqrtAge + sqrtWT + hours_fasting + sex * sterilization_status, data = scDat) 
  out.fit <- lm(MZ ~ krt_cp_bun_value + sqrtAge + sqrtWT + hours_fasting + sex * sterilization_status, data = scDat)
  medModList[[i]] <- mediate(med.fit, out.fit, treat = "sqrtAge", mediator = "krt_cp_bun_value",  sims=10000, boot=T) 
}


names(medModList) <- dogmzs
sensList <- lapply(medModList, function(x) medsens(x))
sumList <- lapply(medModList, function(x) summary(x))
out <- lapply(sumList, medOut)
out <- do.call(rbind, out)
out$mz <- dogmzs
out$ADE_FDR <- p.adjust(out$ADE_P, 'fdr')
out$ACME_FDR <- p.adjust(out$ACME_P, 'fdr')
out$sensitivity <- sapply(sensList, sensitiveTest)
out$mediator <- 'krt_cp_bun_value'

save(out, medModList, sensList, file='dog metabolome paper/mediationByBUN')




rm(list=ls())
################
load('dog metabolome paper/mediationByBUN')
out[out$ACME_FDR<=0.05, ]

dev.off()
plot(medModList[['N-Ac-Phenylalanine']])

# how often are the ptmAAs mediated by creat?
mods <- read.csv('dog metabolome paper/modAAs.csv')
table(mods$type)
out$PTM <- out$mz %in% mods$mz[mods$type=='PTM']

table('mediated'=out$ACME_FDR<=0.05, 'ptmAA'=out$PTM) # but, how many of these ptmAAs are age-associated????
fisher.test(table('mediated'=out$ACME_FDR<=0.05, 'ptmAA'=out$PTM))

a <- read.table('dog metabolome paper/TableS1.txt', sep='\t', head=T)
head(a)
colnames(a)[colnames(a)=='FDR'] <- 'age_FDR'
colnames(a)[colnames(a)=='age.Beta'] <- 'B_age'

head(a)

x <- bind_cols(out[a$Metabolite, ], a[ ,c('Metabolite', 'B_age', 'P.value', 'age_FDR')])


table('mediated'=x$ACME_FDR<=0.05, 'ptmAA'=x$PTM, 'age.assoc'=x$age_FDR<=0.05) # the age effect of 4 of the 7 age-assoc ptmAAs are mediated by bun. whereas none of the 6 non-age-assoc mzs are medaited 
ptmAAsMediated <- print(x$mz[x$ACME_FDR<=0.05 & x$PTM & x$age_FDR<=0.05]) # age-assoc. ptmAAs mediated by bun
x$mz[x$ACME_FDR<=0.05 & x$PTM & x$age_FDR<=0.05 & x$B_age>0]
x$mz[x$ACME_FDR<=0.05 & x$PTM & x$age_FDR<=0.05 & x$B_age<0]

barplot(out$ADEpropOfTotal.Effect[out$mz %in% ptmAAsMediated])
hist(out$ADEpropOfTotal.Effect[out$ACME_FDR<=0.05])


sumList <- lapply(medModList, function(x) summary(x))
p <- lapply(sumList, function(x) data.frame('propMed'=x$n0, 'LCI'=x$n.avg.ci[1], 'UCI'=x$n.avg.ci[2]))
p <- do.call(rbind, p)
p$mz <- rownames(p)

head(p)
head(out)

p <- left_join(p, dplyr::select(out, c(mz, ACME_FDR)))

mediationCandidates <- c("Dimethylarginine (A/SDMA)",  "Hydroxyproline",  "N6-Acetyl-Lysine",  "N6-Trimethyllysine",  "N-Ac-Alanine",  "N-Ac-L-Glutamine",  "N-Ac-Phenylalanine",  "N-Ac-Tryptophan",  "N-Ac-Aspartate",  "N-AcetylGlycine",  "n-Formylmethionine")

head(p)

p <- p[p$mz %in% x$mz[x$age_FDR<=0.05], ] # limit to only those age-assoc. ptmAAs

p$sig <- as.factor(ifelse(p$ACME_FDR>0.05, 'NS', 'sig'))

p$mz[p$mz=="N-Acetyl-Aspartate (NAA)"] <- "N-Ac-Aspartate"
p$mz <- factor(p$mz, levels=p$mz[order(p$propMed)])

bunMedPlot <- ggplot(subset(p, mz %in% mediationCandidates), aes(y=propMed, x=mz, fill=sig))+
  geom_bar(stat='identity', color='black') + 
  geom_errorbar(aes(ymax = UCI, ymin=LCI, width=0.1))+
  theme_bw(base_size = 16)+
  theme(legend.position = 'none')+
  scale_fill_manual(values=c('NS'='white', 'sig'='grey40'))+
  coord_flip(ylim = c(0, 1)) + 
  xlab('')+
  ylab('proportion mediated (+/- 95%CI)')+
  ggtitle('mediation by blood urea nitrogen')

bunMedPlot 

save(bunMedPlot, file='dog metabolome paper/plots/bunMedPlot')


load('dog metabolome paper/plots/bunMedPlot')
load('dog metabolome paper/plots/creatMedPlot')

ggarrange(creatMedPlot + 
            ggtitle('mediation by blood creatinine')  + 
            theme_bw(base_size = 14)+
            theme(legend.position = 'none'),
          bunMedPlot +
            theme_bw(base_size = 14)+
            theme(legend.position = 'none'),
          ncol=1)




rm(list=ls())
###################################
# mediation analysis: urine specific gravity (uSG): 
###################################
load('dog metabolome paper/UrineMixedModelData')
load('dog metabolome paper/UrineMixedModelResults')

sensitiveTest <- function(x) ifelse(x$ind.d0[x$rho==0]==0, 'passes', 'fails')
medOut<- function(x) data.frame('ADE' = x$z0, 'ADElowerCI' = x$z0.ci[1], 'ADEupperCI'= x$z0.ci[2], 'ADE_P' = x$z0.p, 'ACME' = x$d0, 'ACMElowerCI' = x$d0.ci[1], 'ACMEupperCI'= x$d0.ci[2], 'ACME_P' = x$d0.p)

library(mediation)

n.cores <- detectCores(all.tests = F, logical = T)
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  


# mediation 
medModList <- list()
clusterExport(clus, varlist=c('testUvars', 'u', 'dogmzs'), envir = environment()) 

for (i in 1:length(dogmzs)) {
  scu$MZ <- scu[ ,dogmzs[i]]
  med.fit <- lm(MZ ~ sqrtAge + sqrtWT + hours_fasting + sex * sterilization_status + method + volume, data = scu) 
  out.fit <- lm(MZ ~ krt_urine_sg + sqrtAge + sqrtWT + hours_fasting + sex * sterilization_status + method + volume, data = scu)
  medModList[[i]] <- mediate(med.fit, out.fit, treat = "sqrtAge", mediator = "krt_urine_sg",  sims=1000, boot=T) 
}

names(medModList) <- dogmzs
sensList <- lapply(medModList, function(x) medsens(x))
sumList <- lapply(medModList, function(x) summary(x))
out <- lapply(sumList, medOut)
out <- do.call(rbind, out)
out$mz <- dogmzs
out$ADE_FDR <- p.adjust(out$ADE_P, 'fdr')
out$ACME_FDR <- p.adjust(out$ACME_P, 'fdr')
out$sensitivity <- sapply(sensList, sensitiveTest)
out$mediator <- 'krt_urine_sg'
save(out, medModList, sensList, file='mediationByScaledUrineSG')


load('mediationByScaledUrineSG') 
table(out$sensitivity) #

sumList <- lapply(medModList, function(x) summary(x))
p <- lapply(sumList, function(x) data.frame('propMed'=x$n0, 'LCI'=x$n.avg.ci[1], 'UCI'=x$n.avg.ci[2]))
p <- do.call(rbind, p)
p$mz <- rownames(p)

head(p)
head(out)
p <- left_join(p, dplyr::select(out, c(mz, ACME_FDR)))

mediationCandidates <- c("Dimethylarginine (A/SDMA)",  "Hydroxyproline",  "N6-Acetyl-Lysine",  "N6-Trimethyllysine",  "N-Ac-Alanine",  "N-Ac-L-Glutamine",  "N-Ac-Phenylalanine",  "N-Ac-Tryptophan",  "N-Ac-Aspartate",  "N-AcetylGlycine",  "n-Formylmethionine")

p <- p[p$mz %in% mediationCandidates, ]
head(p)

p$sig <- as.factor(ifelse(p$ACME_FDR>0.05, 'NS', 'sig'))
p$mz <- factor(p$mz, levels=p$mz[order(p$propMed)])

ggplot(p, aes(y=propMed, x=mz, fill=sig))+
  geom_bar(stat='identity', color='black') + 
  geom_errorbar(aes(ymax = UCI, ymin=LCI, width=0.1))+
  theme_bw(base_size = 16)+
  theme(legend.position = 'none')+
  scale_fill_manual(values=c('NS'='white', 'sig'='grey40'))+
  coord_flip() + 
  xlab('')+
  ylab('proportion mediated (+/- 95%CI)')+
  ggtitle('mediation by blood urine specific gravity')

# the plot is not very useful as there is onlu one ptmAA assoc. with SG, 
## summarize this effect (on hydroxyproline) in text.
p[p$sig=='sig', ]


