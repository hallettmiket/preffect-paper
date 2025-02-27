knitr::opts_chunk$set(echo = TRUE)
library(genefu); library(tidyverse)
library(anndata)
library(sva); library(pamr); library(Rsubread); library(DESeq2);

# creates the 'pam50' variable; we didn't have to do this before, I wonder what changed
data("pam50", package = "genefu")

# The two BC datasets we don’t have PAM50 subtypes is “GSE47462” and the “Metastatic Breast Cancer Project” datasets
# here, we add them using the same methods used for our DCIS dataset 
# ER/PR/HER2 status for GSE47462 could be useful in evaluating subtype assignments.
library(reticulate)
use_python("/path/to/bin/python")
source("~/path/to/src/gene_sets.R")

# path to files
args <- commandArgs(trailingOnly = TRUE)
# args[1] will be the path you passed from Python
anndata_path <- args[1]


# 3) Read your H5AD file:

adataR <- anndataR::read_h5ad(anndata_path)


# Read an AnnData file
adata <- read_h5ad(anndata_path)

# need to convert ensemble IDs to gene IDs if necessary
ensg_count <- grepl("^ENSG", adata$var_names)  

if (sum(ensg_count) > 1) {
    print("Gene name conversion active.")
    source("~/repo/dcisproject/src/init.R")
    X <- read_rds(file="/path/to/extended_expression_shipment_1_and_2_additional_info.rda")

    test <- sapply(
    adata$var_names, 
    id2name,               # 'i2g' gets gene_name, but you're never passing converting_library
    X[['id2gene']]
    )

    adataR$var_names <- unname(test)
    adata$var_names <- unname(test)
}
# lets look at the data, make sure formatting is okay


expression_matrix <- adata$X
rownames(expression_matrix) <- adata$obs_names
colnames(expression_matrix) <- adata$var_names

original_counts_matrix <- adataR$layers$original_counts
rownames(original_counts_matrix) <- adataR$obs_names
colnames(original_counts_matrix) <- adataR$var_names


obs_metadata <- adata$obs
var_metadata <- adata$var

# Now we work towards getting VST normalization going on these datasets; it uses DeSeq2
obs_metadata_framed <- data.frame(
    batch = factor(obs_metadata$batch)
)

DS <- DESeqDataSetFromMatrix(countData=round(t(expression_matrix)), 
                             colData = obs_metadata_framed, 
                             design = ~  1  )  
DS <- estimateSizeFactors(DS)
VST <- varianceStabilizingTransformation(DS, blind = FALSE)


# And now we do the PAM50 designation

# Find PAM50 genes
rownames(pam50$centroids)[which(rownames(pam50$centroids)=="CDCA1")]<-"NUF2"
rownames(pam50$centroids)[which(rownames(pam50$centroids)=="KNTC2")]<-"NDC80"
rownames(pam50$centroids)[which(rownames(pam50$centroids)=="ORC6L")]<-"ORC6"

my_pam50 <- signatures$basic$pam50$gene_name


idxs <- c(); missing <- c()
for (i in 1:length(my_pam50)) {
   tmp <- which(rownames(VST)==my_pam50[i])
   if (length(tmp)==0) { missing <- c(missing, my_pam50[i])} else   idxs <- c(idxs, tmp)
}
print(missing) # nothing missing without filtering

# if something is missing, we should exclude it from pam50$centroids
if (!is.null(missing)) {
    pam50$centroids <- pam50$centroids[!rownames(pam50$centroids) %in% missing, ]
}

# now we scale VST normalized data 
mat <- assay(VST) %>% t %>% scale %>% t %>% as.matrix

mat <- mat[rownames(pam50$centroids), ]

pearson_corr <- matrix(data=NA, nrow=5, ncol=ncol(mat), dimnames=list(colnames(pam50$centroids)))
spearman_corr <- matrix(data=NA, nrow=5, ncol=ncol(mat), dimnames=list(colnames(pam50$centroids)))


for (i in 1:ncol(mat)) {
   for (j in 1:5) {
     # loops through the expression of PAM50 genes by individual patient
     # computes that correlation between the patient and each 
      pearson_corr[j, i] <- cor(mat[,i], pam50$centroids[,j], method="pearson") 
      spearman_corr[j, i] <- cor(mat[,i], pam50$centroids[,j], method="spearman") 
   }
}
pearson_subtype <- rownames(pearson_corr)[apply(pearson_corr, MARGIN=2, which.max)]
spearman_subtype <- rownames(spearman_corr)[apply(spearman_corr, MARGIN=2, which.max)]

print("First set done")

# and now we add them to the AnnData structure
obs_metadata$subtype_endogenous = pearson_subtype
obs_metadata$subtype_spearman_endogenous = spearman_subtype
adata$obs = obs_metadata


# now I want to replace subtype and subtype_spearman based on the genes in this dataset

# Now we work towards getting VST normalization going on these datasets; it uses DeSeq2
obs_metadata_framed <- data.frame(
    batch = factor(adata$obs$batch)
)

DS2 <- DESeqDataSetFromMatrix(countData=round(t(original_counts_matrix)), 
                             colData = obs_metadata_framed, 
                             design = ~  1  )  

DS2 <- estimateSizeFactors(DS2)

VST <- varianceStabilizingTransformation(DS2, blind = FALSE)



# now we scale VST normalized data 
mat <- assay(VST) %>% t %>% scale %>% t %>% as.matrix


mat <- mat[rownames(pam50$centroids), ]

pearson_corr <- matrix(data=NA, nrow=5, ncol=ncol(mat), dimnames=list(colnames(pam50$centroids)))
spearman_corr <- matrix(data=NA, nrow=5, ncol=ncol(mat), dimnames=list(colnames(pam50$centroids)))

for (i in 1:ncol(mat)) {
    for (j in 1:5) {
        # loops through the expression of PAM50 genes by individual patient
        # computes that correlation between the patient and each 
        pearson_corr[j, i] <- cor(mat[,i], pam50$centroids[,j], method="pearson") 
        spearman_corr[j, i] <- cor(mat[,i], pam50$centroids[,j], method="spearman") 
    }
}
pearson_subtype <- rownames(pearson_corr)[apply(pearson_corr, MARGIN=2, which.max)]
spearman_subtype <- rownames(spearman_corr)[apply(spearman_corr, MARGIN=2, which.max)]

# and now we add them to the AnnData structure
obs_metadata$subtype_2 = pearson_subtype
obs_metadata$subtype_spearman_2 = spearman_subtype
adata$obs = obs_metadata
print("Second Set Done")
print(anndata_path)

write_h5ad(adata, filename = anndata_path)



