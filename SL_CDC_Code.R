# Adapting AN & LB's code to MOMI data 

# load the relevant packages
packages <- c("dplyr","tidyverse","ggplot2","SuperLearner","VIM","recipes","resample","caret","SuperLearner",
              "data.table","nnls","mvtnorm","ranger","xgboost","splines","Matrix","xtable","pROC","arm",
              "polspline","ROCR","cvAUC", "KernelKnn", "gam")
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
# Just creating a smaller data set for coding purposes -- delete later
#D <- D %>% dplyr::select(c(ch_smmtrue, married_X1, married_X99, anesth_re_No, anesth_re_Yes, birthweight, induced_No, induced_Yes))
#D$R1 <- runif(693, -1, 1)
#D$R2 <- runif(693, -10, 10)
#D$R3 <- runif(693, 0, 1)

# Specify the number of folds for V-fold cross-validation
folds=10

#-------------------------------------------------------------------------------
# Hand-coding Super Learner
#-------------------------------------------------------------------------------
## 1: split data into 10 groups for 10-fold cross-validation 
splt<-split(D,1:folds)

splt_splines <- split(D_splines, 1:folds)

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
# can put everything into an xgb.Dmatrix and use xgb.cv?
xgb_dat <- D %>% dplyr::select(-c(ch_smmtrue))
xgb_dat <- as.matrix(xgb_dat)
xgb_label <- D %>% dplyr::select(c(ch_smmtrue))
xgb_label <- as.matrix(xgb_label)
dtrain <- xgb.DMatrix(data =xgb_dat, label = xgb_label)
mx1 <- xgboost(data = xgb_dat, label = xgb_label, eta = 0.1, max_depth = 15, nround = 25, objective="binary:logistic")

mx2 <- xgb.cv(data = xgb_dat, label = xgb_label, objective = "binary:logistic",
              nfold = 10, nround = 25, eta = 0.1, max_depth = 15)

mx2 <- xgb.cv(data = dtrain, objective = "binary:logistic",
              nfold = 10, nround = 25, eta = 0.1, max_depth = 15)
# These both work. I don't know how important it is to do the cross-validation manually.

mxx <- lapply(1:folds, function(ii) xgboost(data=as.matrix(splt[[-ii]][,-11]), label=as.matrix(splt[[-ii]][,11]), eta = 0.1, 
                                           max_depth = 15, nround=25))
# the above is what i originally tried, it doesn't work

#k-nearest-neighbors
mxx <- lapply(1:folds, function(ii) KernelKnn(do.call(rbind,splt[-ii][,-11]), TEST_data = NULL,do.call(rbind,splt[-ii][,11]), k=5))

mxx <- lapply(1:folds, function(ii) knnreg(do.call(rbind,splt[-ii]), do.call(rbind,splt[-ii][,11]), k = 5))
fit <- knnreg(D, D$ch_smmtrue, k = 5)
# Still getting error about incorrect number of dimensions; it's related to use of the extract operator []
# ALso getting an error if I double-bracket ii (splt[[ii]]): attempt to select more than one element in integerOneIndex


#glmnet
mxx <- lapply(1:folds, function(ii) glmnet(do.call(rbind,splt[-ii][,-11]), do.call(rbind,splt[-ii][,11]),  alpha = 0))
# Also getting an incorrect # of dimensions here 
# There's a cv.glmnet option, should I use that? 
mx4 <- glmnet(xgb_dat, xgb_label, family=c("binomial"))
mx5 <- cv.glmnet
# these above both work 

#CDC algorithm



## 2c: obtain the predicted probability of the outcome for observation in the ii-th validation set
p1<-lapply(1:folds,function(ii) predict(m1[[ii]],newdata=rbindlist(splt[ii]),type="response"))
p2<-lapply(1:folds,function(ii) predict(m2[[ii]],data=rbindlist(splt[ii])))

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
