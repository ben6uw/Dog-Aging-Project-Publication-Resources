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


rm(list=ls())
################################################################################################
## quality control, normalization, imputation
################################################################################################
load('data/metabolome/merged.data')

par(mfrow=c(2,2))
boxplot(t(dat), pch=19, cex=0.1, xlab='metabolites', ylab='raw peak area') 
boxplot(dat[ ,1:100], pch=19, cex=0.1, xlab='samples', ylab='raw peak area')

dat <- log(dat)
dat <- scale(dat, scale=F)
dat <- as.data.frame(t(dat))

boxplot(dat, pch=19, cex=0.1, xlab='metabolites', main='log mzs\nscaled by sample') # by mz
boxplot(t(dat[ ,1:100]), pch=19, cex=0.1, xlab='samples', main='log mzs\nscaled by sample') # by samples

dat <- data.frame(samps, dat)
colnames(dat)[!colnames(dat) %in% colnames(samps)] <- panel # trick to thwart the re-coding of colnames for the mzs
dat$group <- as.factor(dat$group)
dat$prep_batch <- as.factor(dat$prep_batch) 
dat$run_order <- as.numeric(dat$run_order)
table(is.na(dat$run_order), dat$group) # run orders recorded for the samples, the interspersed QCs are within the run order
dat$dog_id <- as.character(dat$dog_id)

table(dat$group)
table(dat$graded_hemolysis, dat$group) # hemolysis score for plasma samples
table(dat$prep)
table(dat$cohort)
table(dat$cohort, dat$graded_hemolysis) # hemolysis among the samples from each cohort

# Exclude Hemolyzed Samples (remove score =4)---------panel# Exclude Hemolyzed Samples (remove score =4)--------------------------------------------
dat <- dat[dat$group=='Sample', ]
dat <- dat[dat$grade != 4, ]

# Removing the most- missing metabolites ----------------------------------------------
mzdat <- dat[ ,panel]
table(colSums(is.na(mzdat))) # NA count per metabolite, 108 complete metabolites, 180 completely missing metabolites
mzdat
par(mfrow=c(1,1))
hist(colSums(is.na(mzdat)/nrow(mzdat)), border=0, col='grey40', xlab='missingness (0 to 1)', main='', 30)

K <-print(round(nrow(mzdat) * 0.1)) # 10% of 1274 samples = 127
mzdat <- mzdat[ ,colSums(is.na(mzdat)) <= K] # 10% missingness threshold, no more than 10% missing

dogmzs <- colnames(mzdat) # dog metabolites that pass to this point

### the Raftery lab identified 4 features that were artifacts, spurious detection of other metabolites. remove them:
correlated.artifacts <- c("4-Guanidinobutanoate", "5,6-Dihydrouracil", "Aminolevulinate", "Glyceraldehyde")
dogmzs <- dogmzs[!dogmzs %in% correlated.artifacts]
# these are the dog metabolites that we will work with

mzdat <- mzdat[ ,dogmzs]

dat <- dat[ ,!colnames(dat) %in% panel] # remove mzs from dat
dat <- data.frame(dat, mzdat)

colnames(dat)[!colnames(dat) %in% colnames(samps)] <- dogmzs # trick to thwart the re-coding of colnames

# plots to show batch (prep) and LCMS run order effects:
par(mfrow=c(5,4), mar=c(4,4,1,1)+0.5)

for (i in 1:length(dogmzs)) {
  if(!all(is.na(dat[,dogmzs[i]]))) {
    plot(dat[ ,dogmzs[i]], pch=20, cex=0.5,
         main = dogmzs[i],
         xlab = "", ylab = "Abundance", las = 2, col = dat$prep_batch) +
      abline(v = c(74,229,444,726)) }} # vertical lines delineate LCMS runs

head(dat)
p1 <- ggplot(dat, aes(y=`Trimethylamine-N-Oxide (TMAO)`, x=1:nrow(dat), color=prep_batch))+
  geom_point(size=0.5)+
  xlab('')+
  theme_bw(base_size = 12)+
  geom_smooth(method='lm', se=F)+
  theme(legend.position = 'none')

p2 <- ggplot(dat, aes(y=`Trimethylamine-N-Oxide (TMAO)`, x=1:nrow(dat), color=prep_batch))+
  geom_point(size=0.5)+
  xlab('')+
  theme_bw(base_size = 8)+
  geom_smooth(method='lm', se=F)+
  facet_wrap(~prep_batch, scales='free')+
  theme(legend.position = 'none')

ggarrange(p1, p2)


# correct batch and run order
# then impute missing

table(colSums(is.na(dat[ ,dogmzs]))) # missingness, up to 125 NAs per metabolite, which is below 10% of the samples

table(dat$prep_batch) # seems prep batches are large enough to estimate a linear effect of run order.  
## doing mz ~ run order x prep batch would also better accomdate non-linear run order effects (see valine ~ run order in the last LCMS run = non-linear) 
## this would also simultaneously adjust for batch and LCMS run effects:

batchRunCorrected <- dat[ ,dogmzs]
# this loop keeps the NAs in the residuals
for(i in 1:length(dogmzs)){
  batchRunCorrected[ ,dogmzs[i]] <- residuals(lm(dat[ ,dogmzs[i]] ~ run_order * prep_batch, dat, na.action=na.exclude)) }

# confirm that this saved the position of all NAs
table(colSums(is.na(dat[ ,dogmzs])))
table(colSums(is.na(batchRunCorrected[ ,dogmzs])))
dat[,dogmzs][is.na(batchRunCorrected)] # should all be NA

par(mfrow=c(2,2))
plot(batchRunCorrected$`1-Methylnicotinamide`, dat$`1-Methylnicotinamide`)
plot(batchRunCorrected$Xanthine, dat$Xanthine)

par(mfrow=c(5,4), mar=c(4,4,1,1)+0.5)

for (i in 1:length(dogmzs)) {
  if(!all(is.na(dat[,dogmzs[i]]))) {
    plot(batchRunCorrected[ ,dogmzs[i]], pch=20, cex=0.5,
         main = dogmzs[i],
         xlab = "", ylab = "Abundance", las = 2, col = dat$prep_batch) +
      abline(v = c(74,229,444,726)) }} # vertical lines delineate LCMS runs

tmp <- dat
tmp[ ,dogmzs] <- batchRunCorrected

p3 <- ggplot(tmp, aes(y=`Trimethylamine-N-Oxide (TMAO)`, x=1:nrow(dat), color=prep_batch))+
  geom_point(size=0.5)+
  xlab('')+
  theme_bw(base_size = 12)+
  geom_smooth(method='lm', se=F)+
  theme(legend.position = 'none')

ggarrange(ggarrange(p1, p3), p2, ncol=1, heights = c(0.65, 1))

dat[ ,dogmzs] <- batchRunCorrected

# it looks like there are differences in mz variance among the LCMS runs.
# example TMAO

# it looks like there are differences in mz variance among the prep batches too.
# example pyruvate

# if we scale (DO NOT ALSO CENTER, scale ONLY, see GLUCOSE) by prep batch, it would handle both LCMS run and prep batch
dev.off()

scat <- dat %>% group_by(prep_batch) %>% mutate_at(dogmzs, function(x) c(scale(x, center=F)))


ggplot(scat, aes(y=`Trimethylamine-N-Oxide (TMAO)`, x=1:nrow(dat), color=prep_batch))+
  geom_point(size=0.5)+
  xlab('')+
  theme_bw(base_size = 16)+
  geom_smooth(method='lm', se=F)+
  theme(legend.position = 'none')

p4 <- ggplot(scat, aes(y=`Trimethylamine-N-Oxide (TMAO)`, x=1:nrow(dat), color=prep_batch))+
  geom_point(size=0.5)+
  xlab('')+
  theme_bw(base_size = 16)+
  geom_smooth(method='lm', se=F)+
  theme(legend.position = 'none')

ggarrange(p1, p3, p4, nrow=1)


# no remaining secular trends (nearly an axiom at this point).

# Summarize missingness
miss <- colSums(is.na(scat[ ,dogmzs]))
max(miss)
mean(miss)
max(miss)/nrow(scat)



# Impute
##############################################################
imp <- impute.knn(t(scat[ ,dogmzs]))
imp <- t(imp$data)
imp[1:4,1:4]

for (i in 1:length(dogmzs)) {
  if(!all(is.na(dat[,dogmzs[i]]))) {
    plot(imp[ ,dogmzs[i]], pch=20, cex=0.5,
         main = dogmzs[i],
         xlab = "", ylab = "Abundance", las = 2, col = dat$prep_batch) +
      abline(v = c(74,229,444,726)) }} # vertical lines delineate LCMS runs

dat[ ,dogmzs] <- imp

save(dat, dogmzs, file="data/metabolome/ProcessedData/normalizedData")
###############################
rm(list=ls())




