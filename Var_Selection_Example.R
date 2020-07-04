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

