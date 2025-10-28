library(ggplot2)
library(RColorBrewer)
library(plyr)
library(tidyverse)
library(dplyr)
library(ggpubr)
library(usmap)
library(ggridges)



setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')


rm(list=ls())
#################################################################################################
# "signalment" in the Precision Cohort baseline metabolome
#################################################################################################

load("data/metabolome/ProcessedData/technicalCovarsRemovedData") # metabolome adjusted for technical covariates (shipping and LC-MS)
head(dat)
table(dat$cohort) # remove P2, and P3, merge EDTA with P1
dat$cohort[dat$cohort=='Precision 1 EDTA'] <- 'Precision 1'
dat <- dat[dat$cohort %in% 'Precision 1', ] #865 P1 dogs

save(dat, dogmzs, file='data/metabolome/ProcessedData/P1.technicalCovarsRemovedData')



#################################################################################################
# age, sex, sterilization and other dog status data
#################################################################################################
load('data/2024_release/DAP_2024_DogOverview_v1.0.RData') # load sex sterilization and other meta data
DogOverview <- haven::zap_labels(DogOverview)

DogOverview <- DogOverview[ ,c("dog_id", "Estimated_DOB", "Estimated_Age_Years_at_HLES", "Sex_Class_at_HLES", "Weight_Class_5KGBin_at_HLES", "Weight_Class_10KGBin_at_HLES", "Breed_Size_Class_at_HLES", "LifeStage_Class_at_HLES")]
DogOverview$dog_id <- as.character(DogOverview$dog_id)

dat <- right_join(DogOverview, dat)

dat$Sex_Class_at_HLES
dat <- separate(dat, 'Sex_Class_at_HLES', into=c('sex', 'sterilization_status'), sep=', ')

dat$sex
dat$sex <- as.factor(dat$sex)

dat$sterilization_status <- as.factor(ifelse(dat$sterilization_status == 'intact', 'intact', 'sterile'))
table(dat$sterilization_status, dat$sex)

dat$LifeStage_Class_at_HLES <- as.factor(dat$LifeStage_Class_at_HLES)
table(dat$LifeStage_Class_at_HLES)
levels(dat$LifeStage_Class_at_HLES) <- c('NA', 'Mature', 'Puppy', 'Senior', 'Young')
dat$LifeStage_Class_at_HLES <- factor(dat$LifeStage_Class_at_HLES, levels=c('Puppy', 'Young', 'Mature', 'Senior'))
dat$LifeStage_Class_at_HLES[is.na(dat$LifeStage_Class_at_HLES)]



##################################################################
# load sample metadata 
##################################################################
## metadata about the samples that Maria combed from redcap:
MariasSampleMetaData <- read.csv('dog metabolome paper/data/P1sampleinfo.csv') # not all data is  availabe for all dogs
MariasSampleMetaData$dog_id <- as.character(MariasSampleMetaData$dog_id)
head(MariasSampleMetaData)

MariasSampleMetaData <- select(MariasSampleMetaData, c(dog_id, cohort, weight_at_DOC, lifestage_at_DOC, size_at_DOC, state))


### 2024 data release meta data:
meta <- read.csv('data/2024_release/DAP_2024_SamplesResults_Metadata_v1.0.csv', head=T) # load sample metadata
head(meta)
meta$dog_id <- as.character(meta$dog_id)

# get sample meta data from chem panel or metabolome, they are the same blood draw:
m <- meta[meta$Sample_Year=='precision_1' & meta$Sample_Type=='Chemistry Panel' | meta$Sample_Year=='precision_1' & meta$Sample_Type=='Metabolome', ]
m <- select(m, c(dog_id, Sample_Year, Sample_Collection_DateTime, Sample_Dog_Weight, Sample_Dog_Weight_Units))
table(duplicated(m))
m <- m[!duplicated(m), ]

head(m)
m$DOC <- as.Date(m$Sample_Collection_DateTime) # make 'DOC' date of collection
m <- select(m, -Sample_Collection_DateTime)
head(m)

dat <- right_join(m, dat)
dat <- right_join(MariasSampleMetaData, dat)


# age at DOC: 
dat$Estimated_DOB <- as.Date(dat$Estimated_DOB)
dat$AgeAtDOC <- as.numeric(difftime(dat$DOC, dat$Estimated_DOB))/365

colSums(is.na(dat))


## weight at DOC

table(is.na(dat$weight_at_DOC)) # Maria, who had pulled the most accurate weigths, has some missing data
table(is.na(dat$weight_at_DOC), is.na(dat$Sample_Dog_Weight)) # all but two of the missing data are in the 2024 release
plot(Sample_Dog_Weight ~ weight_at_DOC, dat, col=as.numeric(as.factor(dat$Sample_Dog_Weight_Units)))

dat$Sample_Dog_Weight[dat$Sample_Dog_Weight_Units=='kg' & !is.na(dat$Sample_Dog_Weight) & !is.na(dat$Sample_Dog_Weight_Units)] <- dat$Sample_Dog_Weight[dat$Sample_Dog_Weight_Units=='kg' & !is.na(dat$Sample_Dog_Weight) & !is.na(dat$Sample_Dog_Weight_Units)] * 2.205 # convert all sample dog weights from kg to lbs
dat <- dat %>% mutate(weight_at_DOC = coalesce(weight_at_DOC, Sample_Dog_Weight))
table(is.na(dat$weight_at_DOC)) # only 2 dogs without weight at DOC now.



save(dat, dogmzs, file='data/metabolome/ProcessedData/P1.technicalCovarsRemovedData_metaToo')


rm(list=ls())
#########################################################################
## add genetic breed data
###########################################################################
load('dog metabolome paper/data/p1_update_March2024_with_Genetic_Breeds.RData') # loads new 'dat'
breeds <- dat[ ,1:4]
head(breeds )


load('data/metabolome/ProcessedData/P1.technicalCovarsRemovedData_metaToo')
dat <- right_join(breeds, dat) # genome/breed data not available for at least one P1 dog)



#########################################################################
## add CBC data
#########################################################################
load('dog metabolome paper/data/blood.covariates.RData') # loads 'b', normalized CBC data for the P1 dogs


dat <- right_join(dat, select(b, -c(cohort, elapse_DOC_LOG, serum_temp))) # trims to 785 dogs, one of these dogs does not have genome_breed data


cellVars <- bloodvars[grepl('cbc', bloodvars)]
cellVars <- cellVars[!grepl('rel', cellVars)] # remove partly-redundant/colinear 'rel' relative counts, leave 'abs', absolute counts
cellVars


save(dat, dogmzs, bloodvars, cellVars, file='data/metabolome/ProcessedData/P1.metabolome.withBreedData')




rm(list=ls())

# summarize dog demographics
#######################################################
load('data/metabolome/ProcessedData/P1.metabolome.withBreedData')

## define common breeds:
table(dat$pct>0.6)
breedTable <- table(dat$geneticTopBreed, '85%+ ancestry'=ifelse(dat$pct>=0.85, "85%+", "<85%"))
write.csv(breedTable, quote=F, file='breedDistributionTable.csv') # distribution of breeds at >60% ancestry

table(dat$pct>=0.85)/785

table(dat$sterilization_status)/785

ggplot(dat, aes(x=100*pct))+
  geom_histogram(alpha=0.5, position = 'identity')+
  theme_classic(base_size = 16)+
  ylab('dogs')+
  xlab('top-breed ancestry (%)')

# representation of the top breeds:
head(breedTable)
topbreeds <- breedTable[ ,'85%+'][rev(order(breedTable[ ,'85%+']))][1:20] # the top 20 breeds
topbreeds <- topbreeds[topbreeds >=8] # at least 8 dogs
head(topbreeds)

cnt <- dat[dat$pct>=0.85, ] %>% group_by(geneticTopBreed, sex) %>%  dplyr::summarise(n=n())
head(cnt)

cnt <- cnt[cnt$geneticTopBreed %in% names(topbreeds), ]
cnt$geneticTopBreed <- factor(cnt$geneticTopBreed, levels=names(topbreeds))

ggplot(cnt, aes(geneticTopBreed, n, fill=sex)) + 
  geom_col() +
  theme_classic(base_size = 16) +
  coord_flip()+
  scale_x_discrete(limits=rev)+
  ylab('dogs')+
  xlab('')


sum(cnt$n[cnt$geneticTopBreed != "Newfoundland"])
nrow(dat)-sum(cnt$n[cnt$geneticTopBreed != "Newfoundland"]) # dogs not among the top breeds (excluding Newfoundlands)
cnt$geneticTopBreed

# exclude Newfoundland from breed-level analysis as it is confounded with sex
dat$commonPurebred <- dat$geneticTopBreed
dat$commonPurebred[!dat$geneticTopBreed %in% names(topbreeds)[names(topbreeds)!= "Newfoundland"] | dat$pct <0.85] ='remaining dogs' # threshold for data among the common purebreds
table(dat$commonPurebred)

dat$commonPurebred <- as.factor(dat$commonPurebred)
levels(dat$commonPurebred)

levels(dat$commonPurebred)[levels(dat$commonPurebred)=="Cavalier King Charles Spaniel"] <- "Cav. King Ch. Spaniel"
levels(dat$commonPurebred)[levels(dat$commonPurebred)=="German Shepherd Dog"] <- "German Shepherd"
levels(dat$commonPurebred)[levels(dat$commonPurebred)=="American Pitbull Terrier"] <- "Am. Pitbull Terrier"

table(dat$commonPurebred)
table(dat$commonPurebred != 'remaining dogs')

cnt <- dat %>% group_by(commonPurebred, sex) %>%  dplyr::summarise(n=n())
head(cnt)

x <- dat %>% group_by(commonPurebred) %>% dplyr::summarise(n=n())
head(x)
cnt$commonPurebred <- factor(cnt$commonPurebred, levels=rev(c(x$commonPurebred[order(x$n)])))
head(cnt)

ggplot(subset(cnt, commonPurebred!= 'remaining dogs'), aes(commonPurebred, n, fill=sex)) + 
  geom_col() +
  theme_classic(base_size = 16) +
  coord_flip()+
  scale_x_discrete(limits=rev)+
  ylab('dogs')+
  xlab('')

ggplot(dat, aes(x=AgeAtDOC, fill=sex))+
  geom_histogram(alpha=0.5, position = 'identity')+
  theme_classic(base_size = 18) +
  ylab('dogs')+
  xlab('age (years)')

summary(dat$AgeAtDOC)
range(dat$AgeAtDOC)
table(dat$sex)/nrow(dat)
table(dat$sterilization_status)/nrow(dat)
table(dat$sterilization_status, dat$sex)
median(dat$AgeAtDOC[dat$sex=='Female']) # median female age
median(dat$AgeAtDOC[dat$sex=='Male']) # median male age






cellVars <- bloodvars[grepl('cbc' ,bloodvars)]

save(dat, dogmzs, cellVars, bloodvars, file='data/metabolome/ProcessedData/P1.metabolome.withBreedData.2')





rm(list=ls())
##############################################################################
load('data/metabolome/ProcessedData/P1.metabolome.withBreedData.2')
dat[1:4,1:4]

dat$geneticTopBreed
table(dat$pct >=0.85) # 334 dogs with >85% top breed percentage

pure <- dat[dat$pct>=0.85, ]
plot(table(pure$geneticTopBreed))
hist(table(pure$geneticTopBreed))
table(pure$geneticTopBreed)

table(table(pure$geneticTopBreed) >=8) 

pure <- pure %>%
  group_by(geneticTopBreed) %>%
  filter(n()>= 8)
table(pure$geneticTopBreed)

pure <- pure[pure$geneticTopBreed!='Newfoundland', ]

pure$sex <- as.factor(pure$sex)
levels(pure$sex) <- c('female', 'male')

ggplot(pure, aes(weight_at_DOC, x=geneticTopBreed, color=sex))+
  geom_violin(alpha=0.5, color = "transparent")+
  geom_point(size=1)+
  theme_classic(base_size = 14)+
  coord_flip()

head(pure)

meanW <- aggregate(weight_at_DOC ~ geneticTopBreed, pure, mean)
pure$geneticTopBreed <- factor(pure$geneticTopBreed, levels=rev(meanW$geneticTopBreed[order(meanW$weight_at_DOC)]))

ggplot(pure, aes(weight_at_DOC, y=geneticTopBreed, fill=geneticTopBreed))+
  geom_density_ridges(alpha=0.6, color=NA, scale = 2.5)+
  theme_bw(base_size = 16)+
  theme(legend.position = 'none')+
  xlim(0,100)

# plot the distribution of all reamining dogs
reaminingDogs <- dat[!dat$dog_id %in% pure$dog_id, ]

ggplot(reaminingDogs, aes(weight_at_DOC, fill='grey'))+
  geom_density(alpha=0.6, color=NA)+
  theme_bw(base_size = 16)+
  scale_fill_manual(values='grey')+
  theme(legend.position = 'none')+
  xlim(0,100)+
  xlab('Mixed Breed')


p1 <- ggplot(pure, aes(weight_at_DOC, y=geneticTopBreed, fill=geneticTopBreed))+
  geom_density_ridges(alpha=0.6, color=NA, scale = 2.5)+
  theme_classic(base_size = 16)+
  theme(legend.position = 'none')+
  xlim(0,100)


p2 <- ggplot(reaminingDogs, aes(weight_at_DOC, fill='grey'))+
  geom_density(alpha=0.6, color=NA)+
  theme_classic(base_size = 16)+
  scale_fill_manual(values='grey')+
  theme(legend.position = 'none')+
  xlim(0,100)+
  xlab('Mixed Breed')

p1
ggarrange(p1 +  theme(axis.title.x=element_blank(),axis.text.y=element_blank()), 
          p2+  theme(axis.text.y=element_blank()) + xlab('weight (kg)'),
          ncol=1, heights=c(1, 0.5)) 






#################################################################################################
# PCA
#################################################################################################

rm(list=ls())
##############################################################################
load('data/metabolome/ProcessedData/P1.metabolome.withBreedData.2')

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



cat(paste(cellVars, "+")) # trick to get model terms

library(car) # enables type III anova/ancova

aList <- list()

for(k in 1:sigEigs) {
aList[[k]] <- Anova(aov(lm(dat[ ,pcs[k]] ~ sqrtAge + sqrtWT + lifestage_at_DOC + sex + sterilization_status + hours_fasting + commonPurebred + krt_cbc_total_wbcs + krt_cbc_abs_bands + krt_cbc_abs_neutrophils + krt_cbc_abs_lymphocytes + krt_cbc_abs_monocytes + krt_cbc_abs_eosinophils + krt_cbc_abs_basophils + krt_cbc_rbc + krt_cbc_hgb + krt_cbc_hct + krt_cbc_mcv + krt_cbc_mch + krt_cbc_mchc + krt_cbc_rdw + krt_cbc_mpv + krt_cbc_pct + krt_cbc_retic_abs, dat)), type="III")  }

terms <- rownames(aList[[1]])

p <- as.data.frame(t(sapply(aList, function(x) x$`Pr(>F)`))) # pvalues
rownames(p) <- pcs[1:sigEigs]
colnames(p) <- terms
table(p$sqrtAge <0.05)

ss <- as.data.frame(t(sapply(aList, function(x) x$`Sum Sq`))) # sum of squares
rownames(ss) <- pcs[1:sigEigs]
colnames(ss) <- terms
rowSums(ss)

ss$PC <- rownames(ss)
ss['PC4', ]

## only look at SS among the CBC variables:
ss <- reshape2::melt(ss[ ,c('PC', cellVars)])
head(ss)
colnames(ss)[2:3] <- c('term', 'SS') 
ss$term <- gsub('krt_cbc_', '', ss$term)
ss$PC <- factor(ss$PC, levels=pcs[1:sigEigs])
levels(ss$PC) <- 1:sigEigs
ss$term <- factor(ss$term)
levels(ss$term)
levels(ss$term) <- c("Abs Bands", "Abs Basophils", "Abs Eosinophils", "Abs Lymphocytes", "Abs Monocytes", "Abs Neutrophils", "HCT", "HGB", "MCH", "MCHC", "MCV", "MPV", "PCT", "RBC", "RDW", "Retic Abs Count", "Total WBCs") # simpler names - match names on Table S2

colours=pals::glasbey (nlevels(ss$term))

cbcANCOVAplot <- ggplot(ss, aes(y=SS, x=PC, fill=term))+
  geom_bar(stat='identity')+
  theme_classic(base_size = 16)+
  theme(axis.text.x = element_text(size = 12))+
  scale_fill_manual(values=colours, name = "CBC term")

cbcANCOVAplot

save(cbcANCOVAplot, file='plots/cbcANCOVAplot')


# combine the var by CBC variables
ss <- as.data.frame(t(sapply(aList, function(x) x$`Sum Sq`))) # sum of squares
rownames(ss) <- pcs[1:sigEigs]
colnames(ss) <- terms

ss$CBC <- apply(ss[ ,cellVars], 1, sum) 
ss <- ss[ ,!colnames(ss) %in% cellVars]
ss$PC <- rownames(ss)

ss <- reshape2::melt(ss)
head(ss)
colnames(ss)[2:3] <- c('term', 'SS') 
ss <- ss[ss$term!='(Intercept)', ]
ss$PC <- factor(ss$PC, levels=pcs[1:sigEigs])
levels(ss$PC) <- 1:sigEigs
ss$term <- factor(ss$term)
levels(ss$term)
ss$term <- factor(ss$term, levels=c("sqrtAge", "sqrtWT", "lifestage_at_DOC", "sex", "sterilization_status", "commonPurebred", "hours_fasting", "CBC", "Residuals"))
levels(ss$term) <- c('age', 'weight', 'lifestage', 'sex', 'sterilization', 'breed', 'fasting', 'cbc', 'residuals') # simpler names

nlevels(ss$term)
colors <- c(RColorBrewer::brewer.pal(nlevels(ss$term)-1, 'Set1'), 'grey80')

ggplot(ss, aes(y=SS, x=PC, fill=term))+
  geom_bar(stat='identity')+
  theme_classic(base_size = 16)+
  theme(axis.text.x = element_text(size = 12))+
  scale_fill_manual(values=colors)

ggplot(subset(ss, term!= 'residuals'), aes(y=SS, x=PC, fill=term))+
  geom_bar(stat='identity')+
  theme_bw(base_size = 20)+
  scale_fill_manual(values=colors, '')+
  theme(panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        panel.grid.minor.y = element_blank())


residualANCOVAplot <-  ggplot(ss, aes(y=SS, x=PC, fill=term))+
  geom_bar(stat='identity')+
  theme_classic(base_size = 16)+
  theme(axis.text.x = element_text(size = 12))+
  scale_fill_manual(values=colors, name = "ANCOVA term")


ggarrange(residualANCOVAplot, cbcANCOVAplot)




# PCS ~ common breed to see if PCs show breed-specificity:
pcVars <- paste0('PC', 1:sigEigs)

ANOVAbreedP <- sapply(aList, function(x) x$`Pr(>F)`[7])
breedPCs <- pcVars[ANOVAbreedP<=0.05] # PCS for which breed is significant at P<=5%

breedPCs

l <- pivot_longer(dat[ ,colnames(dat) %in% c(terms, breedPCs)], cols=all_of(breedPCs), names_to='PC')
head(l)
l$PC <- factor(l$PC, levels=breedPCs)
l$commonPurebred <- factor(l$commonPurebred, levels=unique(l$commonPurebred))
l$commonPurebred

ggplot(l, aes(y=value, x=commonPurebred, color=commonPurebred, fill=commonPurebred))+
  geom_boxplot(alpha=0.5, outlier.shape = NA)+
  geom_jitter(width=0.01, size=0.2)+
  theme_classic(base_size = 12)+
  xlab('breed')+
  theme(axis.text.x=element_blank())+
  facet_wrap(~PC, nr=4, scales='free')+
  scale_fill_discrete('breed')+
  scale_color_discrete('breed')
    
# order: 
# breeds by size

l$sqrtWT
meanWT <- aggregate(sqrtWT ~ commonPurebred, l, mean)
meanWT
brwtOrder <- as.character(meanWT$commonPurebred[order(meanWT$sqrtWT)])
l$commonPurebred <- factor(l$commonPurebred, levels=brwtOrder)

ggplot(l, aes(y=value, x=commonPurebred, color=commonPurebred, fill=commonPurebred))+
  geom_boxplot(alpha=0.5, outlier.shape = NA, size=0.2)+
  geom_jitter(width=0.05, size=0.4)+
  theme_classic(base_size = 12)+
  theme(axis.text.x=element_blank())+
  facet_wrap(~PC, nr=1, scales='free')+
  ylab('PC value')+
  xlab('breeds, ordered by mean cohort dog weight')+
  scale_fill_discrete(name = "")+
  scale_color_discrete(name = "")

meanAge <- aggregate(sqrtAge ~ commonPurebred, l, mean)
plot(meanAge$sqrtAge, meanWT$sqrtWT, pch=19)

ggplot(dat, aes(y=PC4, x=sqrtAge))+
  theme_classic(base_size = 20)+
  geom_smooth(method='lm', se=T, size=0) +
  geom_point()+
  xlab(expression("age" ~ sqrt(years)))

## the effect of age on PC4 in a full model:
##NOTE: ran all PCs in model mixed model with GRM below, beyond the ANOVA

ageM <- lm(PC4  ~ sqrtAge + sqrtWT + lifestage_at_DOC + sex + sterilization_status + hours_fasting + commonPurebred + krt_cbc_total_wbcs + krt_cbc_abs_bands + krt_cbc_abs_neutrophils + krt_cbc_abs_lymphocytes + krt_cbc_abs_monocytes + krt_cbc_abs_eosinophils + krt_cbc_abs_basophils + krt_cbc_rbc + krt_cbc_hgb + krt_cbc_hct + krt_cbc_mcv + krt_cbc_mch + krt_cbc_mchc + krt_cbc_rdw + krt_cbc_mpv + krt_cbc_pct + krt_cbc_retic_abs, dat)

summary(ageM)


p1 <- ggplot(subset(ss, term!= 'residuals'), aes(y=SS, x=PC, fill=term))+
  geom_bar(stat='identity')+
  ylab('ANOVA (SS)')+
  theme_classic(base_size = 16)+
  scale_fill_manual(values=colors)+
  theme(strip.text.x = element_blank(),
      strip.background = element_rect(colour="white", fill="white"),
      legend.position=c(0.8,.7))

p2 <- ggplot(dat, aes(y=PC4, x=sqrtAge))+
  geom_smooth(method='lm', se=T, size=0) +
  geom_point(size=0.8)+
  theme_classic(base_size = 16)+
  xlab(expression("age" ~ sqrt(years)))

ggarrange(p1, p2, widths=c(1, 0.9))

save(aList, ss, pcs, terms, sigEigs, file='dog metabolome paper/PCANOVA_result')




#######################################################
load('dog metabolome paper/PCANOVA_result')
ss[ss$term=='breed', ]

### how much of the variance can be explaiend by the signalment variables (1-residual SS) per PCs:
ss <- as.data.frame(t(sapply(aList, function(x) x$`Sum Sq`)))
rownames(ss) <- pcs[1:sigEigs]
colnames(ss) <- terms
ss
varExp <-  t(ss %>% ungroup() %>%
  mutate(across()/rowSums(across())))
round(varExp, 5)
head(varExp)

ss$totalvariance <- apply(ss, 1, sum) 

propVar <- apply(ss, 2, function(x) x/ss$totalvariance)
propVar[1, ]

apply(propVar, 2, max) # the maximum proportion of variance (of PCs) explained by each model term/level 

table(dat$commonPurebred)

ss$CBC <- apply(ss[ ,cellVars], 1, sum) 
ss <- ss[ ,!colnames(ss) %in% cellVars]
head(ss)
hist(ss$CBC/ss$totalvariance)
range(ss$CBC/ss$totalvariance)
mean(ss$CBC/ss$totalvariance)
sum(ss$CBC/ss$totalvariance)

ss$PC <- rownames(ss)
range(1-(ss$Residuals/ss$totalvariance)) # range of the variance explained by all covariates

ss$PC <- factor(ss$PC, levels=pcs[1:sigEigs])
levels(ss$PC) <- 1:sigEigs

ggplot(ss, aes(y=1-(Residuals/totalvariance), PC))+
  geom_bar(stat='identity')+
  theme_classic(base_size = 16)+
  ylab('proportion of variance')

cat(paste0("'",levels(dat$commonPurebred), "',"))

dat$commonPurebred <- factor(dat$commonPurebred, levels=c('Cav. King Ch. Spaniel', 'French Bulldog', 'Border Collie', 'Poodle', 'Golden Retriever', 'Labrador Retriever', 'German Shepherd', 'Great Dane', 'remaining dogs'))

breedcolors <- c(RColorBrewer::brewer.pal(12, name='Paired'), 'grey70')
breedcolors <- c("#A6CEE3", "#1F78B4", "#B2DF8A", "#33A02C", "#FB9A99", "#E31A1C", "#FDBF6F", "#FF7F00", "#CAB2D6", "#6A3D9A", "#f542d4", "#B15928", "grey70") 

ggplot(dat, aes(y=PC2, x=sqrtWT, color=commonPurebred))+
  geom_point()+
  theme_classic(base_size = 16)+
  geom_smooth(method='lm', se=F) +
  scale_color_manual(values=breedcolors)+
  labs(color="breed")+
  xlab(expression("weight" ~ sqrt(kg)))


save(PCs, sigEigs, file='dog metabolome paper/mzPCs')


## PC snp heritability: variance in metabolome PCs explained by the relatedness in the grm:
rm(list=ls())
###################################

load('dog metabolome paper/mzPCs')
load('dog metabolome paper/data/scaled.data_for_CBC_mixedModel') # from age_mixed_model_CBCcovars.R

PCs <- merge(scDat, PCs)
table(PCs$dog_id %in% rownames(grm))
PCs <- PCs[PCs$dog_id %in% rownames(grm), ] # g is now composed of variables that have been scaled
grm <- grm[PCs$dog_id, PCs$dog_id] # subset and order the GRM by the dogs in the mz data

###################################
# fit a mixed model, with fixed effects of age, sex, weight, etc, in the context of random effects of relatedness (as represented in the GRM)

Zmat <- diag(nrow(PCs)) # an empty design matrix for the random effects

# build a design matrix for the fixed effects; specify interaction terms here
vars[!vars%in%dogmzs]

tmp <- ' ~ sqrtAge + sqrtWT + sex * sterilization_status + hours_fasting'
cat(c(tmp, paste0('+ ', cellCovars))) # manually copy and paste this into the line below.

X <- model.matrix( ~ sqrtAge + sqrtWT + sex * sterilization_status + hours_fasting + krt_cbc_total_wbcs + krt_cbc_abs_bands + krt_cbc_abs_neutrophils + krt_cbc_abs_lymphocytes + krt_cbc_abs_monocytes + krt_cbc_abs_eosinophils + krt_cbc_abs_basophils + krt_cbc_rbc + krt_cbc_hgb + krt_cbc_hct + krt_cbc_mcv + krt_cbc_mch + krt_cbc_mchc + krt_cbc_rdw + krt_cbc_mpv + krt_cbc_pct + krt_cbc_retic_abs, data=PCs) 

pcVars <- paste0('PC', 1:sigEigs)
pcDat <- PCs[ ,pcVars]

###################################
# the next steps can be slow, use parallel processing
library(parallel)
library(doParallel)
library(EMMREML)

n.cores <- detectCores(all.tests = F, logical = T) # detects the number of available cores
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  

emma <- emmreml(y=pcDat[ ,1], X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) # the model fitting step, run an example to see what it does and to record the output rownames etc. (the code below will call the output)
emma$Vu/(emma$Vu+emma$Ve) # variance in Y explained by relatedness
 

###################################
# this code fits the same model twice, the first time it extracts the fixed effects, and the second time, it takes the random effects (they are called BLUPs):

# get fixed effects
clus <- makeCluster(n.cores) 
registerDoParallel(cores=n.cores)  
clusterExport(clus, varlist=c("pcDat", 'pcVars', 'X', 'Zmat', 'grm'), envir = environment()) 

mList <- list()

for(i in 1:length(pcVars)) {
  clusterExport(clus, varlist=c("pcDat", 'pcVars', 'X', 'Zmat', 'grm'), envir = environment()) 
  Y <- pcDat[ ,i]
  mList[[i]] <- emmreml(y=Y, X=X, Z=Zmat, K=grm, varbetahat = T, varuhat=T, PEVuhat=T, test=T) }


save(mList, PCs, pcVars, X, sigEigs, file='dog metabolome paper/PCMixedModel_results')




rm(list=ls())
##########################################################
# heritability of PCs
##########################################################
load('dog metabolome paper/PCMixedModel_results')

h <- sapply(mList, function(x) x$Vu/sum(x$Vu+x$Ve))
PCherit <- data.frame(PC=pcVars, heritability=h)
head(PCherit)

mean(PCherit$heritability)
range(PCherit$heritability)

PCherit$PC <- factor(PCherit$PC, levels=pcVars)
levels(PCherit$PC) <- 1:sigEigs

ggplot(PCherit, aes(x = PC, y=heritability)) +
  geom_bar(stat="identity") +
  theme_bw(base_size = 20) +
  labs(y= expression(H["SNP"]))+
  xlab('Principal Component')
  

load('dog metabolome paper/PCANOVA_result')
head(ss)

nlevels(ss$term)
colors <- c(RColorBrewer::brewer.pal(nlevels(ss$term)-1, 'Set1'), 'grey80')

anovaPlot  <- ggplot(subset(ss, term!= 'residuals'), aes(y=SS, x=PC, fill=term))+
  geom_bar(stat='identity')+
  ylab('ANOVA (SS)')+
  xlab('Principal Component')+
  theme_bw(base_size = 20) +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank()
  )+
  scale_fill_manual(values=colors, name = "")+
  theme(strip.text.x = element_blank(),
        strip.background = element_rect(colour="black", fill=NA),
        legend.position=c(0.8,.63), legend.text=element_text(size=16))

hsnpPlot <- ggplot(PCherit, aes(x = PC, y=heritability)) +
  geom_bar(stat="identity") +
  theme_bw(base_size = 20) +
 theme(
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank()
      )+
  labs(y= expression(H["SNP"]))+
  xlab('Principal Component')


ggarrange(anovaPlot, NULL, hsnpPlot,   ncol = 1, heights=c(1, 0.05, 0.7))
  

mean(PCherit$heritability) # report
range(PCherit$heritability) # report











