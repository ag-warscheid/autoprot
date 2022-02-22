CRANpkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE, repos='https://ftp.fau.de/cran/')
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

BCpkgTest <- function(x){
  installedPackges <- installed.packages() %>%
    rownames
  if (x %in% installedPackges){
    require(x,character.only = TRUE)
  }
  else {
    BiocManager::install(x)
    require(x,character.only = TRUE)
  }
}



args = commandArgs(trailingOnly=TRUE)
func <- args[1]
input <- args[2]
output <- args[3]

#specifically for lm
#design <- args[5]

# dfv <- read.table("D:\\data\\Wignand\\Notebooks\\Python\\HD\\LargeExperiment/test.csv",
#                     sep='\t', header=TRUE)
CRAN_packages <- c("rrcovNA", "tidyverse", "BiocManager")
for (package in CRAN_packages){
  CRANpkgTest(package)
}
BC_packages <- c("limma", "vsn", "RankProd")
for (package in BC_packages){
  BCpkgTest(package)
}
df <- read.table(input, sep='\t', header=TRUE)
rownames(df) <- df$UID
dfv <- df[,-which(names(df) %in% c("UID"))]

#----- functions

impSeqFunction <- function(dfv) {
  dfv <- as.data.frame(impSeq(dfv))
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
}

limmaFunction <- function(dfv) {
  
  design <- args[4]
  if (design == "twoSample") {
    design <- read.table(args[5], sep='\t', header=TRUE)
  }
  else if (design == "custom") {
    design <- read.table(args[5], sep='\t', header=TRUE)
  }
  else{
    coef <- rep(1,ncol(dfv))
    design <- data.frame(coef)
  }
  
  fit <- lmFit(dfv, design)
  eb <- eBayes(fit)
  res <- topTable(eb, coef="coef", number=Inf, confint = TRUE)
  res$UID <- rownames(res)
  write.table(res, output, sep='\t')
}

quantileNorm <- function(dfv) {
  dfv <- normalizeBetweenArrays(dfv, method="quantile")
  dfv <- as.data.frame(dfv)
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
}

vsnNorm <- function(dfv) {
  dfv <- normalizeVSN(dfv)
  dfv <- as.data.frame(dfv)
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
}

cyclicLOESS <- function(dfv) {
  dfv <- normalizeCyclicLoess(dfv)
  dfv <- as.data.frame(dfv)
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
}

rankprod <- function(dfv) {
  data.cl.sub <- c(rep(c(1), NCOL(dfv)))
  data.origin.sub <- rep(1,NCOL(dfv))
  data.sub <- as.matrix(dfv)
  
  RP.out <- RP.advance(data.sub, data.cl.sub, data.origin.sub,logged=TRUE,
                       na.rm=TRUE, gene.names=df$`UID`, plot=FALSE, calculateProduct = FALSE,
                       rand=1337)
  
  #plotRP(RP.out, cutoff=0.05)
  
  RS <- RP.out$RSs[,2]
  logFC <- RP.out$AveFC
  pval <- RP.out$pval
  pfp <- RP.out$pfp
  
  res <- data.frame(RS, logFC, pval, pfp)
  res$UID <- rownames(res)
  
  colnames(res) = c("RS", "logFC","PValue_class1<class2","PValue_class1>class2",
                    "adj.P.Val_class1<class2","adj.P.Val_class1>class2","UID")
  write.table(res, output, sep='\t')
}

#------ switchCase

result = switch(
  func,
  "impSeq"   = impSeqFunction(dfv),
  "limma"    = limmaFunction(dfv),
  "quantile" = quantileNorm(dfv),
  "vsn"      = vsnNorm(dfv),
  "cloess"   = cyclicLOESS(dfv),
  "rankProd" = rankprod(dfv)
)
