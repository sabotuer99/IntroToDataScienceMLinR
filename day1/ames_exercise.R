# load packages
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(magrittr)
  

# load the data DONT NEED USED RSTUDIO DATASET IMPORT
train_set = read.csv("data/ames_housing_train.csv")
train_set$PID <- 1L

# the task
task = TaskRegr$new(id = "ames_housing", backend = train_set, target = "SalePrice")

computeFactorMultiplier = function(x, col, target) {
  j = aggregate(x[,target] ~ x[,col], data = x, mean)
  row.names(j) <- j[,1]
  j[x[,col], 2] / mean(x[,target])
}


#transformation function
transform <- function(x, param_set) { 
  x$PID <- 1
  x$MS.SubClass <- as.factor(x$MS.SubClass)
  x$Bedroom.AbvGr <- as.factor(x$Bedroom.AbvGr)
  x$Kitchen.AbvGr <- as.factor(x$Kitchen.AbvGr)
  x$TotRms.AbvGrd <- as.factor(pmin(x$TotRms.AbvGrd, 11)) # very strong linear relationship from 0 - 10, 11+ gets weird
  x$Bsmt.Full.Bath <- as.factor(x$Bsmt.Full.Bath)
  x$Bsmt.Half.Bath <- as.factor(x$Bsmt.Half.Bath)
  x$Full.Bath <- as.factor(x$Full.Bath)
  x$Half.Bath <- as.factor(x$Half.Bath)
  x$Fireplaces <- as.factor(x$Fireplaces)
  x$Garage.Cars <- as.factor(x$Garage.Cars)
  x$Overall.Cond <- as.factor(x$Overall.Cond)
  x$Overall.Qual <- as.factor(x$Overall.Qual)
  x$Yr.Sold <- as.factor(x$Yr.Sold)
  x$Mo.Sold <- as.factor(x$Mo.Sold)
  
  qx <- round(quantile(x$Year.Built, probs = seq(0,1,0.10)))
  x$Year.Built <- cut(x$Year.Built, qx, include.lowest = TRUE)
  
  qx <- round(quantile(x$Year.Remod.Add, probs = seq(0,1,0.20)))
  x$Year.Remod.Add <- cut(x$Year.Remod.Add, qx, include.lowest = TRUE)
  
  qx <- round(quantile(x$Garage.Yr.Blt, probs = seq(0,1,0.20)))
  x$Garage.Yr.Blt <- cut(x$Garage.Yr.Blt, qx, include.lowest = TRUE)
  x 
  
  cspm = function(col) { computeFactorMultiplier(x, col, "SalePrice") }
  
  # add multiplier metrics for factors
  faccols <- unlist(lapply(x, is.factor))
  facnames = names(x[,faccols])
  for (faccolname in facnames){
    newcolname <- paste0(faccolname, ".Mult")
    newcolnames <- c(colnames(x), newcolname)
    newcol = cspm(faccolname)
    x <- cbind(x, newcol)
    colnames(x) <- newcolnames
  }
  
  x
}

lin_transform = function(x, param_set) {
  trans = transform(x, param_set)
  numcols <- unlist(lapply(trans, is.numeric)) 
  trans = trans[,numcols]
  na.aggregate(trans)
}

knn_transform = function(x, param_set) {
  cols = c(#"Gr.Liv.Area",
           "X1st.Flr.SF",
           "Overall.Qual",
           #"X2nd.Flr.SF",
           #"Total.Bsmt.SF",
           #"Garage.Area",
           "Year.Built",
           #"Fireplaces",
           #"Bsmt.Qual",
           #"Garage.Yr.Blt",
           #"Lot.Area",
           #"Garage.Cars",
           #"Exter.Qual",
           "TotRms.AbvGrd",
           #"Year.Remod.Add",
           #"Full.Bath",
           #"MS.SubClass",
           #"MS.Zoning",
           #"BsmtFin.SF.1",
           #"Garage.Finish",
           #"Kitchen.Qual",
           "Neighborhood")
  trans = transform(x, param_set)[,cols]
  trans
}

lintask = TaskRegr$new(id = "ames_housing_lin", backend = lin_transform(train_set, NULL), target = "SalePrice")

# create a learner
#learner_featureless = lrn("regr.featureless")
learner_ranger <- LearnerRegrRanger$new()
learner_xgboost <- lrn("regr.xgboost", booster = "gblinear")
learner_svm = LearnerRegrSVM$new()
learner_knn = lrn("regr.kknn", k = 15)
learner_glm = lrn("regr.glmnet")

learner_knn$param_set$trafo <- knn_transform
learner_xgboost$param_set$trafo <- transform
learner_ranger$param_set$trafo <- transform
learner_svm$param_set$trafo <- lin_transform
learner_glm$param_set$trafo <- lin_transform

# train the learner
# EXCHANGE THIS PART WITH A PROPER BENCHMARK ANALYSIS!
#learner_featureless$train(task)
learner_ranger$train(task)
#learner_cart$train(task)
#learner_knn$train(task)

# tuning
hout <- rsmp("holdout")
measure <- msr("regr.mse")
evals20 <- term("evals", n_evals = 1000)

ps <- learner_ranger$param_set$clone(deep = TRUE)
ps$subset(c("num.trees", "min.node.size", "importance", "splitrule", "scale.permutation.importance"))
ps$params[1]$num.trees$upper <- 2000
ps$params[2]$min.node.size$upper <- 1000
ps$params[3]$importance$default <- "none"

instance = TuningInstance$new(
  task = task,
  learner = learner_ranger,
  resampling = hout,
  measures = measure,
  param_set = ps,
  terminator = evals20
)
tuner = tnr("gensa", resolution = 5)
result = tuner$tune(instance)
learner_ranger$param_set$values = instance$result$params


at = AutoTuner$new(
  learner = learner_ranger,
  resampling = hout,
  measures = measure,
  tune_ps = ps,
  terminator = evals20,
  tuner = tuner
)
at
at$train(task)
  

# predict on new data
# CHOOSE THE BEST PERFOMING LEARNER OF YOUR BENCHMARK TO PREDICT ON NEW DATA!
bm_design = benchmark_grid(task = task, resamplings = rsmp("cv", folds = 10), 
                           learners = list(
                             learner_ranger, 
                             learner_xgboost,
                             learner_knn
                             ))
bmr = benchmark(bm_design)
performances = bmr$aggregate(list(msr("regr.mse")))
performances[, c("learner_id", "regr.mse")]

bm_design = benchmark_grid(task = lintask, resamplings = rsmp("cv", folds = 10), 
                           learners = list(
                             learner_svm, 
                             learner_glm
                           ))
bmr = benchmark(bm_design)
performances = bmr$aggregate(list(msr("regr.mse")))
performances[, c("learner_id", "regr.mse")]

test_set = read.csv("data/ames_housing_test.csv")
pred = learner_ranger$predict_newdata(task, test_set)

# we can save the predictions as data.table and export them for Kaggle
pred = as.data.table(pred)
pred$truth = NULL
write.csv(pred, "data/ames_housing_submission.csv", row.names = FALSE)


  importance = as.data.table(learner_ranger$importance(), keep.rownames = TRUE)
  colnames(importance) = c("Feature", "Importance")
  importance
