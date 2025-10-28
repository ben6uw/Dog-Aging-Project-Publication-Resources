library(tidyverse)
library(dplyr)
library(ggplot2)
library(plyr)
library(pheatmap)

setwd('/Users/ben/Library/CloudStorage/GoogleDrive-brharrison2000@gmail.com/My Drive/Documents/DogAgingProject')

rm(list=ls())

load('dog metabolome paper/CBC_CELLcovars_mixed_model_results')
load('dog metabolome paper/mmResiduals')

rm(BLUEs)
rm(randomEffects)

betas <- fixedEffects[ ,grepl('beta', colnames(fixedEffects))]
pees <- fixedEffects[ ,grepl('P_', colnames(fixedEffects))]
fdr <- as.data.frame(apply(pees, 2, function(x) p.adjust(x, 'fdr')))
colnames(fdr) <- gsub('P_', 'FDR_', colnames(fdr))
colSums(fdr <=0.05)

a <- data.frame('mz' = rownames(betas), 'ageB'= betas$`beta_sqrtAge`, 'FDR' = fdr$`FDR_sqrtAge`, 'P' = pees$P_sqrtAge)
head(a)

# get modAAs and their parents
n <- print(dogmzs[grepl('N-Ac', dogmzs)])
m <- print(dogmzs[grepl('Dime', dogmzs)])
n6 <- print(dogmzs[grepl('6', dogmzs)])

dogmzs
moi <- c(n, m, n6, '1/3-Methylhistidine', 'Pyroglutamic Acid', 'S-Methylcysteine', 'Hydroxyproline', 'n-Formylmethionine', 'Methionine Sulfoxide')
moi <- moi[!moi %in% c('N-Acetylneuraminate', '5,6-Dihydrouracil', 'N2,N2-Dimethylguanosine', 'G6P')]
moi

dogmzs

ambiguous <- c('1/3-Methylhistidine', 'Pyroglutamic Acid', 'Dimethylglycine', 'N-Ac-Glutamate', 'N-Acetyl-Aspartate (NAA)')
PTM <- moi[!moi %in% ambiguous]
parents <- c('Alanine', 'Glutamic acid', 'Glutamine', 'Phenylalanine', 'Tryptophan', "Aspartic Acid",   'Glycine', 'Arginine', 'Lysine', 'Histidine', 'Proline', 'Methionine') # no Cysteine, S-Methylcysteine has no parent


mp <- data.frame('mz'=c(moi, parents))
mp$type <- NA
mp$type[mp$mz %in% PTM] <- 'PTM'
mp$type[mp$mz %in% ambiguous] <- 'ambiguous'
mp$type[mp$mz %in% parents] <- 'parent'

head(mp)

write.csv(mp, file='dog metabolome paper/modAAs.csv', quote=F, row.names = F)

head(mp)
a <- join(a, mp)

head(a)
a$type[is.na(a$type)] <- 'other'
a$type <- as.factor(a$type)
a$type

a$type <- factor(a$type, levels=c("parent", "ambiguous", 'PTM', "other"))

table(a$type)
table(a$type, ifelse(a$ageB>0, 'pos', 'neg'), 'test'=ifelse(a$FDR<=0.05, 'FDR<5%', 'NS')) # summary of the age-associated metabolites, modAAs and the up/down effect


####################################################################################
## does B-age of the AAs predict the B-age of the corresponding modAA?
head(a)

a$AminoAcid <- tolower(a$mz) 
a$AminoAcid
a$AminoAcid <- gsub("glutamic acid", "glutamate", a$AminoAcid)
a$AminoAcid <- gsub("aspartic acid", "aspartate", a$AminoAcid)
a$AminoAcid <- gsub("n-ac-l-", "", a$AminoAcid)
a$AminoAcid <- gsub("n-ac-", "", a$AminoAcid)
a$AminoAcid <- gsub("n-acetyl-", "", a$AminoAcid)
a$AminoAcid <- gsub("n-acetyl", "", a$AminoAcid)
a$AminoAcid <- gsub("1/3-methyl", "", a$AminoAcid)
a$AminoAcid <- gsub("n6-acetyl-", "", a$AminoAcid)
a$AminoAcid <- gsub("n6-trimethyl", "", a$AminoAcid)
a$AminoAcid <- gsub("dimethyl", "", a$AminoAcid)
a$AminoAcid <- gsub("n-formyl", "", a$AminoAcid)
a$AminoAcid <- gsub("n-acetyl", "", a$AminoAcid)
a$AminoAcid <- gsub("n-formyl", "", a$AminoAcid)
a$AminoAcid <- gsub("s-methyl", "", a$AminoAcid)
a$AminoAcid <- gsub("hydroxy", "", a$AminoAcid)
a$AminoAcid <- gsub("pyro", "", a$AminoAcid)
a$AminoAcid <- gsub(" \\(naa)", "", a$AminoAcid)
a$AminoAcid <- gsub("arginine \\(a/sdma)", "arginine", a$AminoAcid)

a$AminoAcid
table(a$AminoAcid)

head(a)
AAs <- a$AminoAcid[a$type=='PTM']
tmp <- a[a$AminoAcid %in% AAs, ]
tmp <- tmp[tmp$type %in% c('parent', 'PTM'), ]
tmp$type <- droplevels(as.factor(tmp$type))

head(tmp)
w <- tmp[tmp$mz != 'N6-Trimethyllysine' ,-c(1)] %>% pivot_wider(values_from=c(ageB, FDR, P), names_from = type) # manual add trimethyl lysine back to the data, it shares unmodified lysine with acetyl-lysine
w
w <- tmp[tmp$mz != 'N6-Acetyl-Lysine' ,-c(1)] %>% pivot_wider(values_from=c(ageB, FDR, P), names_from = type) # manual add trimethyl lysine back to the data, it shares unmodified lysine with acetyl-lysine

summary(lm(ageB_PTM ~ ageB_parent, w)) # neither N6-Ac-lys, nor trimethyl Lys makes this significant
cor.test(w$ageB_parent, w$ageB_PTM)

ggplot(w, aes(y=ageB_PTM, x=ageB_parent))+
  geom_point()+
  theme_classic(base_size = 16)+
  geom_smooth(method='lm')+
  ylab(expression("ptmAA"~(beta[age])))+
  xlab(expression("AA"~(beta[age])))

table(a$type, a$FDR <=0.05)
table(w$FDR_parent<=0.05, w$FDR_PTM<=0.05)
fisher.test(table(w$FDR_parent<=0.05, w$FDR_PTM<=0.05)) # the ptmAAs assoc. with age are not associated with the parent AAs that are
####################################################################################

a$mzLabel <- a$mz
a$mzLabel[a$type=='other'] <- ''

ggplot(a, aes(ageB, -log(P, 10), color=type, label=mzLabel))+
  geom_point(size=2)+
  theme_classic(base_size = 14)+
  scale_color_manual("", values = c("parent" = "blue", "PTM" = 'red', 'ambiguous'='purple', 'other'='grey80'))+
  ggrepel::geom_text_repel(size=3, max.overlaps = 30,  force = 10, label.padding=10)+
  geom_abline(intercept=1.3, slope=0, linetype='dashed', alpha=0.2)+
  xlab(expression(beta ~ age))+  
  labs(y=expression(-log[10]~P))


# correlation among modAAs and AAs
load('dog metabolome paper/data/blood.covariates.RData') # load CBC covariate names (bloodvars), and the data (b)
b$dog_id <- as.character(b$dog_id)
b <- b[b$cohort=='Precision 1', ]
rownames(b) <- b$dog_id

rownames(mmResiduals) <- scDat$dog_id
mmResiduals$BUN <- b[rownames(mmResiduals), ]$krt_cp_bun_value
mmResiduals$CREAT <- b[rownames(mmResiduals), ]$krt_cp_creatinine_value

colnames(mmResiduals)

rmat <- Hmisc::rcorr(as.matrix(mmResiduals[ ,c(mp$mz)]))$r
pmat <- Hmisc::rcorr(as.matrix(mmResiduals[ ,c(mp$mz)]))$P
fdr <- apply(pmat, 2, function(x) p.adjust(x, 'fdr'))
fdr[rmat<0] = 1 # to measure only positive correlations
colSums(fdr<=0.05, na.rm=T)
mean(colSums(fdr<=0.05, na.rm=T))
range(colSums(fdr<=0.05, na.rm=T)) # number of other metabolites correlated with each AA

rm(pmat)
rm(fdr)

pmat <- Hmisc::rcorr(as.matrix(mmResiduals[ ,c(mp$mz, 'BUN', 'CREAT')]))$P
bunP <- pmat[ ,'BUN']
creP <- pmat[ ,'CREAT']
                  
p <- as.data.frame(pmat[ ,c('BUN', 'CREAT')])
p <- p[!rownames(p) %in% c('BUN', 'CREAT'), ]
head(p)
p$mz <- rownames(p)
p <- merge(p, mp)           
names(p)[names(p)=="BUN"] <- 'corP.BUN'
names(p)[names(p)=="CREAT"] <- 'corP.CREAT'

cMat <- cor(mmResiduals[ ,c(mp$mz, 'BUN', 'CREAT')])
diag(cMat) <- 0

bc <- as.data.frame(cMat[ ,c('BUN', 'CREAT')])
bc <- bc[!rownames(bc) %in% c('BUN', 'CREAT'), ]
bc$mz <- rownames(bc)
rownames(bc) %in% mp$mz
bc$type[mp$mz] <- mp$type
bc <- merge(bc, p)
head(bc)

bc$type <- factor(bc$type, levels=c('parent', 'PTM', 'ambiguous'))
levels(bc$type)[2] <- 'ptm'

table(bc$type, p.adjust(bc$corP.BUN, 'bonferroni') <=0.05, bc$BUN>0)
table(bc$type, p.adjust(bc$corP.CREAT, 'bonferroni') <=0.05, bc$CREAT>0)

head(bc)
bc$sig <- as.factor(ifelse(p.adjust(bc$corP.BUN, 'bonferroni') <=0.05 | p.adjust(bc$corP.CREAT, 'bonferroni') <=0.05, 19, 1))
table(bc$sig)

bc$mz

ggplot(bc, aes(y=CREAT, x=BUN, label=mz, color=type, shape=sig))+
  geom_point(key_glyph = "rect")+
  theme_classic(base_size = 16)+
  ggrepel::geom_text_repel(key_glyph = "rect", size=3)+
  labs(color='modification')+
  scale_color_manual(values=c('grey20', 'red', 'violet'))+
  scale_shape_manual(values=c(1, 19))+
  ylab(expression("blood creatinine (Pearson's"~italic(r)~")"))+
  xlab(expression("blood urea nitrogen (Pearson's"~italic(r)~")"))





bc
bc %>% group_by(type) %>% summarise_at(vars(BUN, CREAT), list(mean_Cor= mean)) # mean person correlation by aa type

table(bc$type)
table(bc$type, 'positively cor'=bc$BUN>0, 'significant'=bc$corP.BUN<=0.05)
table(bc$type, 'positively cor'=bc$CREAT>0, 'significant'=bc$corP.CREAT<=0.05)

rg <- max(abs(cMat))
paletteLength <- 100
myColor <- colorRampPalette(c('purple', "white", 'darkorange'))(paletteLength)
# length(breaks) == length(paletteLength) + 1
# use floor and ceiling to deal with even/odd length pallettelengths
myBreaks <- c(seq(min(cMat), 0, length.out=ceiling(paletteLength/2) + 1), seq(max(cMat)/paletteLength, max(cMat), length.out=floor(paletteLength/2)))
plot(myBreaks)
altBreaks <- c(seq(-max(cMat), 0, length.out=ceiling(paletteLength/2) + 1), seq(max(cMat)/paletteLength, max(cMat), length.out=floor(paletteLength/2)))
plot(altBreaks)

mp$type

anno <- data.frame('class'=as.factor(c(mp$type)))
anno
anno$class
anno$class <- factor(anno$class, levels=c("parent", "ambiguous", 'PTM'))
levels(anno$class) <- c("parentAA", "ambiguous", 'ptmAA')
 
# simplify names for plot margins
rownames(cMat)[rownames(cMat)=="N-Acetyl-Aspartate (NAA)"] <- "N-Ac-Aspartate"
rownames(cMat)[rownames(cMat)=="Dimethylarginine (A/SDMA)"] <- "Dimethylarginine"
colnames(cMat)[colnames(cMat)=="N-Acetyl-Aspartate (NAA)"] <- "N-Ac-Aspartate"
colnames(cMat)[colnames(cMat)=="Dimethylarginine (A/SDMA)"] <- "Dimethylarginine"
rownames(cMat)[rownames(cMat)=="N-AcetylGlycine"] <- "N-Ac-Glycine"
rownames(cMat)[rownames(cMat)=="N6-Acetyl-Lysine"] <- "N6-Ac-Lysine"
colnames(cMat)[colnames(cMat)=="N-AcetylGlycine"] <- "N-Ac-Glycine"
colnames(cMat)[colnames(cMat)=="N6-Acetyl-Lysine"] <- "N6-Ac-Lysine"

rownames(cMat)
cMat <- cMat[!rownames(cMat) %in% c("BUN", "CREAT"), !colnames(cMat) %in% c("BUN", "CREAT")] 

rownames(anno) <- rownames(cMat)
colors <- list(class = c(parentAA='#00A5E3', ambiguous = '#FF96C5',  ptmAA='red'))
myColor <- colorRampPalette(c('orange', "white", 'purple'))(paletteLength)


pheatmap(cMat, color=myColor, clustering_method = "average", breaks=altBreaks, annotation_row = anno, annotation_colors = colors, treeheight_col=0, border_color = 0)



save(a, file='gwas paper/a')



rm(list=ls())
#############################################
load('gwas paper/a')

head(a)
a$type <- as.character(a$type)

## get all of the AAs, parent or not:
AAs <- a$AminoAcid[a$type=='parent']
AAs <- c(AAs, 'asparagine', 'iso-leucine /allo-isoleucine', 'leucine /d-norleucine', 'serine', 'threonine', 'tyrosine', 'valine')

AAs
table(!a$type %in% c('PTM', 'ambiguous') & a$AminoAcid %in% AAs)
a$type[which(!a$type %in% c('PTM', 'ambiguous') & a$AminoAcid %in% AAs)] <- 'AA'
table(a$type)

a$type[which(a$type=='ambiguous')] <- 'other'
table(a$type)


a$type <- as.factor(a$type)
levels(a$type)
levels(a$type) <- c('AA (19)', 'other (102)', 'ptmAA (12)')
a$type <- factor(a$type, c('other (102)', 'AA (19)', 'ptmAA (12)'))

library(ggridges)

ggplot(a, aes(x=ageB, y=type, fill=type, color = PTM)) +
  geom_density_ridges(alpha = 0.5, color=NA)+
  geom_vline(xintercept=0, linetype='dashed', color='grey40')+
  theme_bw(base_size = 16)+
  theme(legend.position = 'none')+
  xlab(expression(beta*" age"))+
  ylab('')+
  scale_fill_manual(values=c('grey', '#00A5E3',  'red'))+
  scale_y_discrete(expand = expansion(add = c(0.2, 1.7)))+
  xlim(c(-0.38, 0.46))





