# load the relevant packages
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam","glmnet")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

#-----------------------------------------------------------------------------------------------------------------------------------------------
# Option 2: simulate data?----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
n=1000
sigma <- abs(matrix(runif(25,0,1), ncol=5))
sigma <- forceSymmetric(sigma)
sigma <- as.matrix(nearPD(sigma)$mat)
x <- rmvnorm(n, mean=c(0,.25,.15,0,.1), sigma=sigma)
modelMat<-model.matrix(as.formula(~ (x[,1]+x[,2]+x[,3]+x[,4]+x[,5])^3))
beta<-runif(ncol(modelMat)-1,0,1)
beta<-c(2,beta) # setting intercept
mu <- 1-plogis(modelMat%*%beta) # true underlying risk of the outcome
y<-rbinom(n,1,mu)

hist(mu);mean(y)

x<-data.frame(x)
D<-data.frame(x,y)
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


# Specify the number of folds for V-fold cross-validation
folds=5
## split data into 5 groups for 5-fold cross-validation 
## we do this here so that the exact same folds will be used in 
## both the SL fit with the R package, and the hand coded SL
index<-split(1:1000,1:folds)
splt<-lapply(1:folds,function(ind) D[index[[ind]],])
# view the first 6 observations in the first [[1]] and second [[2]] folds
head(splt[[1]])
head(splt[[2]])

# Specify the number of folds for V-fold cross-validation
folds=5
## split data into 5 groups for 5-fold cross-validation 
## we do this here so that the exact same folds will be used in 
## both the SL fit with the R package, and the hand coded SL
index<-split(1:1000,1:folds)
splt<-lapply(1:folds,function(ind) D[index[[ind]],])
# view the first 6 observations in the first [[1]] and second [[2]] folds
head(splt[[1]])
head(splt[[2]])


#--------------------------------------------------------------------------------------------------------------------------------------------------------------
# Hand-coding Super Learner -----------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------
## 1: split data into 10 groups for 10-fold cross-validation 
splt<-split(D,1:folds)

splt_splines <- split(D_splines, 1:folds)

## 2: Fitting individual algorithms on the training set (but not the ii-th validation set)
set.seed(123)
#bayesglm
m1<-lapply(1:folds,function(ii) bayesglm(formula=ch_smmtrue~.,data=do.call(rbind,splt[-ii]),family="binomial")) #bayesglm
#glm
m15 <- lapply(1:folds, function(ii) glm(ch_smmtrue~., data=do.call(rbind,splt[-ii]), family="binomial"))
#gam
m16 <- lapply(1:folds,function(ii) gam(ch_smmtrue~., family="binomial",data=rbindlist(splt_splines[-ii])))
#glmnet
m44 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 0, family="binomial"))
m49 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 1.0, family="binomial"))

#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------


#SuperLearner source code for screen.corP
screen.corP <- function(Y, X, family, obsWeights, id, method = 'pearson',
                        minPvalue = 0.1, minscreen = 2)
{
  listp <- apply(X, 2, function(x, Y, method) {
    ifelse(var(x) <= 0, 1, cor.test(x, y=Y, method = method)$p.value)
  },
  Y = Y, method = method)
  whichVariable <- (listp<= minPvalue)
  if(sum(whichVariable)<minscreen){
    warning('number of variables with p value less than minPvalue is less than minscreen')
    whichVariable[rank(listp)<=minscreen] <- TRUE
  }
  return(whichVariable)
}


#SuperLearner source code for screen.corRank
screen.corRank <- function(Y,X,family, method='pearson', rank=2){
  listp <- apply(X,2,function(x,Y,method){
    ifelse(var(x) <= 0, 1, cor.test(x,y=Y, method = method)$p.value)
  }, Y=Y, method=method)
  whichVariable <- (rank(listp)<= rank)
  return(whichVariable)
}


#SuperLearner source code for screen.randomForest
screen.randomForest <- function(Y, X, family, nVar = 10, ntree = 1000, 
                                mtry = ifelse(family$family == "gaussian", floor(sqrt(ncol(X))),
                                              max(floor(ncol(X)/3), 1)),
                                nodesize = ifelse(family$family == "gaussian",5,1), maxnodes = NULL){
  .SL.require('randomForest')
  if(family$family == "gaussian"){
    rank.rf.fit <- randomForest::randomForest(Y~., data = X, ntree = ntree, mtry = mtry, 
                                              nodesize = nodesize, keep.forest = FALSE,
                                              maxnodes = maxnodes)
  }
  if(family$family == "binomial"){
    rank.rf.fit <- randomForest::randomForest(as.factor(Y)~., data = X,
                                              ntree = ntree, mtry = mtry,
                                              nodesize = nodesize, keep.forest = FALSE,
                                              maxnodes = maxnodes)
  }
  whichVariable <- (rank(-rank.rf.fit$imporance)<=nVar)
  return(whichVariable)
}

