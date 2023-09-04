## R Functions for autoprot
## @author: Wignand, Julian

# Function to test installation status and install packages from CRAN
CRANpkgTest <- function(x)
{
  if (!require(x,character.only = TRUE))
  {
    install.packages(x,dep=TRUE, repos='https://ftp.fau.de/cran/')
    if(!require(x,character.only = TRUE)) stop("Package not found")
  }
}

# Function to test installation status and install packages from Bioconductor
BCpkgTest <- function(x){
  installedPackges <- rownames(installed.packages())
  if (x %in% installedPackges){
    require(x,character.only = TRUE)
  }
  else {
    BiocManager::install(x)
    require(x,character.only = TRUE)
  }
}

# Function to test installation status of the archived imputation package required
# for DIMA
ArchpkgTest <- function(){
  installedPackges <- rownames(installed.packages())
  if ("imputation" %in% installedPackges){
    require("imputation",character.only = TRUE)
  }
  else {
    install.packages("https://cran.r-project.org/src/contrib/Archive/imputation/imputation_1.3.tar.gz",
                     repos=NULL,
                     type='source')
  }
}

# Function to test installation status from github/devtools
DTpkgTest <- function(x){
  installedPackges <- rownames(installed.packages())
  pkgName <- strsplit(x, "/")[[1]][[2]]
  if (pkgName %in% installedPackges){
    require(pkgName,character.only = TRUE)
  }
  else {
    devtools::install_github(x)
  }
}

## DEPENDENCIES ##
# autoprot depends on these R packages and will test their installation on runtime
CRAN_packages <- c("rrcovNA", "tidyverse", "BiocManager", "devtools", "glue")
for (package in CRAN_packages){
  CRANpkgTest(package)
}

BC_packages <- c("limma", "vsn", "RankProd", "pcaMethods", "impute", "SummarizedExperiment")
for (package in BC_packages){
  BCpkgTest(package)
}

ArchpkgTest()

devtool_packages <- c("cran/DMwR", "kreutz-lab/DIMAR")
for (package in devtool_packages){
  DTpkgTest(package)
}

## LIBS
library(glue)

## ARGS
# read the args coming form Python
args = commandArgs(trailingOnly=TRUE)

# The functest argument only executes the installation tests and is
# called e.g. during the checkRinstall routine in Python
if (args[1] == 'functest') {
  quit()
}

# RFunctions is called with at least three arguments
func <- args[1]
input <- args[2]
output <- args[3]
# args[4]
# limma:  design/kind of test for limma
# rankProd: class labels of the samples
# DIMA:   list of column headers for comparison of t statistic (joined with ;)
# args[5]
# limma:  location of design file for limma
# DIMA:   methods to use for imputations benchmark
# args[6]
# limma: Comparison strings for contrast calculation
# DIMA:   npat -> number of patterns to simulate
# args[7]
# DIMA:   Performance Metric to use for selection of best algorithm

## READ DATA
# Data for processing is written to file from Python and
# read here for R processing
df <- read.table(input, sep='\t', header=TRUE)
# set the row names of the df to the UID columns
rownames(df) <- df$UID
# remove the column UID from the df and save as new var dfv
dfv <- df[,-which(names(df) %in% c("UID"))]

## FUNCTIONS
# Data driven imputation selection DIMA
dimaFunction <- function(dfv) {

  # default args
  methods <- strsplit(x = args[5], split = ',')[[1]]
  npat <- args[6]
  group <- strsplit(x = args[4], split = ',')[[1]]
  performanceMetric <- args[7]

  mtx <- as.matrix(dfv)
  # from here: copied from
  # https://github.com/kreutz-lab/DIMAR/blob/b8e9497cd490a464bf46ebc177efd7e34f064723/R/dimar.R
  
  if (!group[1] == 'cluster') {
    groupidx <- rep(0L,ncol(mtx))
    groupidx[grepl(group[1],colnames(mtx))] <- 1
    groupidx[grepl(group[2],colnames(mtx))] <- 2
    group <- groupidx
  }
  
  mtx <- DIMAR::dimarMatrixPreparation(mtx, nacut = 2)

  coef <- DIMAR::dimarLearnPattern(mtx)
  ref <- DIMAR::dimarConstructReferenceData(mtx)
  sim <- DIMAR::dimarAssignPattern(ref, coef, mtx, npat)
  
  Imputations <- DIMAR::dimarDoImputations(sim, methods)
  Performance <- DIMAR::dimarEvaluatePerformance(Imputations, ref, sim, performanceMetric, TRUE, group)
  Imp <- DIMAR::dimarDoOptimalImputation(mtx, rownames(Performance))

  dfv <- as.data.frame(Imp)
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
  write.table(Performance, paste(str_sub(output, end=-5), '_performance.csv', sep=""), sep='\t')
}

# sequential imputation from rrcovNA
impSeqFunction <- function(dfv) {
    dfv <- as.data.frame(impSeq(dfv))
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
}

# Linear Models Analysis with limma
limmaFunction <- function(dfv) {
  
  # the design is supplied during call
  design <- args[4]
  # two sample LIMMA
  if (design == "twoSample") {
    design <- read.table(args[5], sep='\t', header=TRUE)
  }
  else if (design == "custom") {
    design <- read.table(args[5], sep='\t', header=TRUE)
  }
  else{
    # Calculation of mean and STDERR of data points
    coef <- rep(1,ncol(dfv))
    design <- data.frame(coef)
    print(design)
  }

  # Test if the matrix is full rank
  if (is.fullrank(design) == FALSE) {
    glue::glue('Matrix is not full rank!')
    glue::glue('The following coefficients cannot be estimated:')
    print(nonEstimable(design))
  }

  # Fit linear model for each protein
  fit <- lmFit(dfv, design)
  
  if (args[6] != "") {
    glue::glue(args[6])
    contrast <- limma::makeContrasts(contrasts=args[6],levels=design)
    fit2 <- limma::contrasts.fit(fit, contrast)
    eb <- eBayes(fit2)
    
    # Extract a table of the top-ranked genes from a linear model fit.
    res <- topTable(eb, coef=args[6], number=Inf, confint = TRUE)
  }
  else {
    # Compute moderated t-statistics, moderated F-statistic, and log-odds
    # of differential expression by empirical Bayes moderation of the standard
    # errors towards a global value.
    eb <- eBayes(fit)
    
    # Extract a table of the top-ranked genes from a linear model fit.
    res <- topTable(eb, coef="coef", number=Inf, confint = TRUE)
  }

  # add back the UID column from the row index
  res$UID <- rownames(res)
  # write out the table
  write.table(res, output, sep='\t')
}

quantileNorm <- function(dfv) {
  dfv <- normalizeBetweenArrays(dfv, method="quantile")
  dfv <- as.data.frame(dfv)
  dfv$UID <- rownames(dfv)
  write.table(dfv, output, sep='\t')
}

vsnNorm <- function(dfv) {
  dfv <- limma::normalizeVSN(dfv)
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
  data.cl.sub <-  unlist(strsplit(x = args[4], split = ','))
  data.origin.sub <- rep(1,NCOL(dfv))
  data.sub <- as.matrix(dfv)
  
  RP.out <- RP.advance(data.sub, data.cl.sub, data.origin.sub, logged=TRUE,
                       na.rm=TRUE, gene.names=df$`UID`, plot=FALSE, calculateProduct=FALSE,
                       rand=1337)
  
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

## SWITCH
# this witch directs the programme flow to one of the target functions depending
# on which statement is provided from within Python
result = switch(
  func,
  "dima"     = dimaFunction(dfv),
  "impSeq"   = impSeqFunction(dfv),
  "limma"    = limmaFunction(dfv),
  "quantile" = quantileNorm(dfv),
  "vsn"      = vsnNorm(dfv),
  "cloess"   = cyclicLOESS(dfv),
  "rankProd" = rankprod(dfv)
)
