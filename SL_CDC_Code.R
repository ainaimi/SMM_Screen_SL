# Adapting AN & LB's code to MOMI data 

# load the relevant packages
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam","glmnet")
for (package in packages) {
  if (!require(package, character.only=T, quietly=T)) {
    install.packages(package,repos='http://lib.stat.cmu.edu/R/CRAN') 
  }
}

#Set working directory
setwd("\\\\136.142.117.70\\Studies$\\Bodnar Abby\\Severe Maternal Morbidity\\Data")

# Read data
D <- readRDS("baked_train_momi_20200615.rds")
D_splines <- readRDS("baked_train_momi_withsplines.rds")
D_knn <- D %>% mutate(smm_knn = ch_smmtrue + 1) %>% dplyr::select(c(-ch_smmtrue))
# Just creating a smaller data set for coding purposes -- delete later
#D <- D %>% dplyr::select(c(ch_smmtrue, married_X1, married_X99, anesth_re_No, anesth_re_Yes, birthweight, induced_No, induced_Yes))
#D$R1 <- runif(693, -1, 1)
#D$R2 <- runif(693, -10, 10)
#D$R3 <- runif(693, 0, 1)

# Specify the number of folds for V-fold cross-validation
folds=10

#-------------------------------------------------------------------------------
# Hand-coding Super Learner
#------------------------------------------------------------------------------2-
## 1: split data into 10 groups for 10-fold cross-validation 
splt<-split(D,1:folds)

splt_splines <- split(D_splines, 1:folds)

splt_knn <- split(D_knn, 1:folds)

## 2: Fitting individual algorithms on the training set (but not the ii-th validation set)
set.seed(123)
# bayesglm with defaults
m1<-lapply(1:folds,function(ii) bayesglm(formula=ch_smmtrue~.,data=do.call(rbind,splt[-ii]),family="binomial")) #bayesglm
#random forest (ranger) with a range of tuning parameters
m2 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = T))
m3 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 3, min.node.size = 10, replace = T))
m4 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 4, min.node.size = 10, replace = T))
m5 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 2, min.node.size = 10, replace = F))
m6 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 3, min.node.size = 10, replace = F))
m7 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 500, mtry = 4, min.node.size = 10, replace = F))
m8 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 2, min.node.size = 10, replace = T))
m9 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 3, min.node.size = 10, replace = T))
m10 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 4, min.node.size = 10, replace = T))
m11 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 2, min.node.size = 10, replace = F))
m12 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 3, min.node.size = 10, replace = F))
m13 <- lapply(1:folds, function(ii) ranger(ch_smmtrue~., data=do.call(rbind,splt[-ii]), num.trees = 2500, mtry = 4, min.node.size = 10, replace = F))
#mean
m14 <- lapply(1:folds,function(ii) mean(rbindlist(splt[-ii])$ch_smmtrue))
#glm
m15 <- lapply(1:folds, function(ii) glm(ch_smmtrue~., data=do.call(rbind,splt[-ii])))
#gams - make sure to use splt_splines 
m16 <- lapply(1:folds,function(ii) gam(ch_smmtrue~., family="binomial",data=rbindlist(splt_splines[-ii])))

#xgboost
m17 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=200, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m18 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=200, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m19 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=200, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m20 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=500, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m21 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=500, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m22 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=500, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m23 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=1000, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m24 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=1000, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m25 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=1000, shrinkage=0.01, nrounds = 15, objective = "binary:logistic", verbose = 0))
m26 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=200, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m27 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=200, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m28 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=200, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m29 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=500, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m30 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=500, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m31 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=500, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m32 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=1000, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m33 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=1000, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m34 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=1000, shrinkage=0.001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m35 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=200, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m36 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=200, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m37 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=200, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m38 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=500, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m39 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=500, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m40 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=500, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m41 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 4, ntrees=1000, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m42 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 5, ntrees=1000, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))
m43 <- lapply(1:folds, function(ii) xgboost(data = as.matrix(do.call(rbind,splt[-ii])[,-11]),label = as.matrix(do.call(rbind,splt[-ii])[,11]), max_depth = 6, ntrees=1000, shrinkage=0.0001, nrounds = 15, objective = "binary:logistic", verbose = 0))

#glmnet - also vary lambdas? 
#cv.glmnet will find the optimal lambda for you... can you incorporate cross-validation of lambda into CV we're already doing 
m44 <- lapply(1:folds, function(ii) cv.glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 0, family="binomial"))
m45 <- lapply(1:folds, function(ii) glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 0.2))
m46 <- lapply(1:folds, function(ii) glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 0.4))
m47 <- lapply(1:folds, function(ii) glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 0.6))
m48 <- lapply(1:folds, function(ii) glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 0.8))
m49 <- lapply(1:folds, function(ii) glmnet(as.matrix(do.call(rbind,splt[-ii])[,-11]), as.matrix(do.call(rbind,splt[-ii])[,11]), alpha = 1.0))

# k-neaest neighbords 
m50 <- lapply(1:folds, function(ii) KernelKnn(do.call(rbind,splt_knn[-ii])[,-196], TEST_data = NULL, as.numeric(as.character(unlist(do.call(rbind,splt_knn[-ii])[,196]))), Levels = unique(as.numeric(as.character(unlist(do.call(rbind,splt_knn[-ii])[,196])))), k=5))

#CDC algorithm: this will just go below because we can skip straight to binary classification 

#-------------------------------------------------------------------------------------------------------------------------------------------
## 2c: obtain the predicted probability of the outcome for observation in the ii-th validation set

#bayesglm - i don't understand this prediction object; list of 10
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=rbindlist(splt[ii]),type="response"))
#ranger - here, it's a list, with predictions in each list object: p2[[1]]$predictions
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],data=rbindlist(splt[ii])))

#mean - m14. below doesn't work because mean is just a list of numbers. assign mean value for every observation in each fold. 
#folds 1-3 have 70 observations; folds 4-10 have 69 observations
p14 <- lapply(1:folds, function(ii) rep(m14[[ii]], nrow(splt[[ii]])))

#glm
# I removed the double brackets from splt[[ii]] and now the command works... 
p15 <- lapply(1:folds, function(ii) predict(m15[[ii]], data = rbindlist(splt[ii])))

#gams - m16
# per predict.Gam documentation, if I take out the data argument and just indicate type, this works. I don't know if the result makes sense, though.
# there has to be a way to supply data (fold) 
p16 <- lapply(1:folds, function(ii) predict(m16[[ii]], newdata=rbindlist(splt_splines[ii]), type="response"))

#xgboost - m17, below doesn't work: Item 1 of input is not a data.frame, data.table, or list
# single bracket around splt[ii] gets rid of that error but now: xgb.DMatrix does not support construction from list
# putting as.matrix around the second argument fixes(?) that error, but now "feature names stored in 'object' and 'newdata' are different!
# this is because of ch_smmtrue (outcome) in splt[[ii]]
# i need to make sure the names *and order* in the object and splt are the same... remove column 11 and now it works 
p17 <- lapply(1:folds, function(ii) predict(m17[[ii]], as.matrix(rbindlist(splt[ii])[,-11]))) 

#glmnet - m35, below doesn't work: "need to supply a value for newx"
# did the same thing as above because "newx" must be a matrix. Now: cholmod error 'X and/or Y have wrong dimensions'
# if I similarly take out column 11 from splt[[ii]], this also works 
# now I don't understand the p44 object really...
p44 <- lapply(1:folds, function(ii) predict(m44[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-11])))
p44 <- lapply(1:folds, function(ii) predict(m44[[ii]], newx = as.matrix(rbindlist(splt[ii])[,-11]), s="lambda.min", type="response"))
p44 <- lapply(1:folds, function(ii) predict(m44[[ii]], newx = as.matrix(do.call(rbind,splt[ii])[,-11]),s="lambda.min",type="response"))# weird output here is because cross validation
#you're seeing predictions for different values of lambda. we're supposed to pick an optimal value of lambda. typically picked with cross-validation.
#

#knn - m41. doesn't work. prediction for knn is weird
#  no applicable method for 'predict' applied to an object of class "c('matrix', 'double', 'numeric')" 
p50 <- lapply(1:folds, function(ii) predict(m50[[ii]], newdata=rbindlist(splt_knn[ii])))

# update dataframe 'splt' so that column1 is the observed outcome (y)
#   column2 is the CV-predicted probability of the outcome from bayesglm
#   column3 is the CV-predicted probability of the outcome from random forest
for(i in 1:folds){
  splt[[i]]<-cbind(splt[[i]][,11],p1[[i]],p2[[i]]$predictions)
}
# view the first 6 observations in the first fold 
head(data.frame(splt[[1]]))

## 2d: calculate CV risk for each method for the ii-th validation set
# our loss function is the rank loss; so our risk is (1-AUC)
#		use the AUC() function with input as the predicted outcomes and 'labels' as the true outcomes
risk1<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,2], labels=splt[[ii]][,1]))    # CV-risk for bayesglm
risk2<-lapply(1:folds,function(ii) 1-AUC(predictions=splt[[ii]][,3], labels=splt[[ii]][,1]))		# CV-risk for knn

#----------------------------
## 3: average the estimated 5 risks across the folds to obtain 1 measure of performance for each algorithm
a<-rbind(cbind("bayesglm",mean(do.call(rbind, risk1),na.rm=T)),
         cbind("polymars",mean(do.call(rbind, risk2),na.rm=T)))
# output a table of the CV-risk estimates
# xtable(a)


#----------------------------
## 4: estimate SL weights using the optim() function to minimize (1-AUC)
X<-data.frame(do.call(rbind,splt),row.names=NULL);  names(X)<-c("y","bayesglm","poly")
head(X)

bounds = c(0, Inf)
SL.r<-function(A, y, par){
  A<-as.matrix(A)
  names(par)<-c("bayesglm","poly")
  predictions <- crossprod(t(A),par)
  cvRisk <- 1 - AUC(predictions = predictions, labels = y)
}
init=(rep(1/2,2))
fit <- optim(par=init, fn=SL.r, A=X[,2:3], y=X[,1], 
             method="L-BFGS-B",lower=bounds[1],upper=bounds[2])
fit
alpha<-fit$par/sum(fit$par)
alpha

#---------------------
## 5a: fit all algorithms to original data
m1<-bayesglm(formula=ch_smmtrue~.,data=D,family="binomial")
m2<-ranger(ch_smmtrue~., data=D)

## 5b: predict probabilities from each fit using all data
p1<-predict(m1,newdata=D,type="response")  # bayesglm
p2<-(predict(m2, data=D))$predictions #randomForest
predictions<-cbind(p1,p2)
head(predictions)

## 5c: for the observed data take a weighted combination of predictions using nnls coeficients as weights
y_pred <- predictions%*%alpha
p<-data.frame(y=D$ch_smmtrue,y_pred=y_pred)

## #--------------------------------------------
# verify that our work predicts similar results as SL package
a<-roc(p$y, p$y_pred, direction="auto")
C2<-data.frame(sens=a$sensitivities,spec=a$specificities)
head(C2)

###--------------------------------------------
# fits from candidate algorithms
a<-roc(D$ch_smmtrue, p1, direction="auto")
Cbayes<-data.frame(sens=a$sensitivities,spec=a$specificities)

a<-roc(D$ch_smmtrue, p2, direction="auto") 
Cpoly<-data.frame(sens=a$sensitivities,spec=a$specificities)

cols <- c("Bayes GLM"="green", "PolyMARS"="black")
ggplot() +
  geom_step(data=Cbayes, aes(1-spec,sens,color="Bayes GLM"),linetype=2,size=.5) +
  geom_step(data=Cpoly, aes(1-spec,sens,color="PolyMARS"),linetype=2,size=.5) +
  #scale_colour_manual(name="",values=cols) +
  theme_light() + theme(legend.position=c(.8,.2)) +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()) + 
  labs(x = "1 - Specificity",y = "Sensitivity") +
  geom_abline(intercept=0,slope=1,col="gray") +
  scale_colour_manual(name="",values=cols)
