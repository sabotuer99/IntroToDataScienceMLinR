# load packages
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(magrittr)
library(ggplot2)
future::plan("multiprocess")

setwd("/home/troywhorten/intro_ds/day2")
train_set = read.csv("data/ames_housing_train_numeric.csv")

# the task
task = TaskRegr$new(id = "ames_housing", backend = train_set, target = "SalePrice")
trans_set = transform(train_set, NULL)

train.pca <- prcomp(trans_set[,names(trans_set) != "SalePrice"], center = TRUE, scale = TRUE)

test.pca <- scale(transform(test_set), center= train.pca$center)
test.pca <- test.pca %*% train.pca$rotation

summary(train.pca)

train.pca.set <- as.data.frame(cbind(train.pca$x, train_set$SalePrice))

pca.task = TaskRegr$new(id = "ames_housing", backend = train.pca.set, target = "V34")

# transform data
transform <- function(x, param_set) { 
  # remove useless columns
  x$PID <- NULL
  x$Low.Qual.Fin.SF <- NULL
  x$X3Ssn.Porch <- NULL
  x$Pool.Area <- NULL
  
  # clip outliers
  x$Lot.Area <- pmin(x$Lot.Area, 25000) 
  x$Lot.Frontage <- pmin(x$Lot.Frontage, 200) 
  x$BsmtFin.SF.1 <- pmin(x$BsmtFin.SF.1, 2500) 
  x$Total.Bsmt.SF <- pmin(x$Total.Bsmt.SF, 3000) 
  x$X1st.Flr.SF <- pmin(x$X1st.Flr.SF, 3000) 
  x$Gr.Liv.Area <- pmin(x$Gr.Liv.Area, 4000) 
  x$Bsmt.Full.Bath <- pmin(x$Bsmt.Full.Bath, 2) 
  x$Bsmt.Half.Bath <- pmin(x$Bsmt.Half.Bath, 1) 
  x$Full.Bath <- pmin(x$Full.Bath, 3) 
  x$Bedroom.AbvGr <- pmin(x$Bedroom.AbvGr, 6) 
  x$Kitchen.AbvGr <- pmax(1, pmin(x$Kitchen.AbvGr, 2))
  x$TotRms.AbvGrd <- pmin(x$TotRms.AbvGrd, 11)
  x$Fireplaces <- pmin(x$Fireplaces, 2)
  x$Garage.Cars <- pmin(x$Garage.Cars, 3)
  x$Wood.Deck.SF <- pmin(x$Wood.Deck.SF, 800)
  x$Open.Porch.SF <- pmin(x$Open.Porch.SF, 400)
  x$Enclosed.Porch <- pmin(x$Enclosed.Porch, 400)
  x$Screen.Porch <- pmin(x$Screen.Porch, 300)
  
  x
}
learner_ranger <- LearnerRegrRanger$new()
#learner_ranger$param_set$trafo <- transform

# clone the ranger params and set some bounds/defaults
rsmp_cv <- rsmp("cv")
measure <- msr("regr.mae")
evals <- term("evals", n_evals = 200)

learner_ranger$param_set$values$splitrule <- "variance"
learner_ranger$param_set$values$scale.permutation.importance <- FALSE 

ps <- learner_ranger$param_set$clone(deep = TRUE)
ps$subset(c("num.trees", "importance"))
ps$params[1]$num.trees$upper <- 300
ps$params[2]$importance$levels <- c("none", "permutation", "impurity")

tuner = tnr("random_search")
  
at = AutoTuner$new(
    learner = learner_ranger,
    resampling = rsmp_cv,
    measures = measure,
    tune_ps = ps,
    terminator = evals,
    tuner = tuner
  )
at$param_set$trafo = transform
at$train(task)
  
test_set = read.csv("data/ames_housing_test_numeric.csv")
pred = at$predict_newdata(task, test_set)
  
# we can save the predictions as data.table and export them for Kaggle
pred = as.data.table(pred)
pred$truth = NULL
write.csv(pred, "data/ames_housing_submission_ranger.csv", row.names = FALSE)

ggplot(at$archive(unnest = "params"),
       aes(x = min.node.size, y = regr.mae, color = scale.permutation.importance)) + geom_point()


# lets do a knn version
learner_knn = lrn("regr.kknn")
kps <- learner_knn$param_set$clone(deep = TRUE)
kps$subset(c("k", "distance", "kernel"))
kps$params[1]$k$upper <- 25
kps$params[2]$distance$upper <- 3

ktuner = tnr("random_search")
kevals <- term("evals", n_evals = 500)

kat = AutoTuner$new(
  learner = learner_knn,
  resampling = rsmp_cv,
  measures = measure,
  tune_ps = kps,
  terminator = kevals,
  tuner = ktuner
)
learner_knn$param_set$trafo = function (x, param_set){
  d = as.data.frame(cbind(x[,1:10], x[,34]))
  d
}
kat$train(pca.task)

ggplot(kat$archive(unnest = "params"),
       aes(x = distance, y = regr.mae, color = scale)) + geom_point()

# lets do a svm version
learner_svm = lrn("regr.svm")
svmps <- learner_svm$param_set$clone(deep = TRUE)
svmps$subset(c("type", "kernel", "degree", "cost"))
svmps$params[2]$kernel$levels <- c("radial", "linear")
svmps$params[3]$degree$upper <- 10
svmps$params[4]$cost$upper <- 3

svmtuner = tnr("random_search")
svmevals <- term("evals", n_evals = 20)

svmat = AutoTuner$new(
  learner = learner_svm,
  resampling = rsmp_cv,
  measures = measure,
  tune_ps = svmps,
  terminator = svmevals,
  tuner = svmtuner
)
svmat$param_set$trafo = transform
svmat$train(task)

# we can save the predictions as data.table and export them for Kaggle
test_set = read.csv("data/ames_housing_test_numeric.csv")
pred = svmat$predict_newdata(task, test_set)
pred = as.data.table(pred)
pred$truth = NULL
write.csv(pred, "data/ames_housing_submission_svm.csv", row.names = FALSE)

ggplot(svmat$archive(unnest = "params"),
       aes(x = degree, y = regr.mae, color = kernel)) + geom_point()


# boosting
  learner_xgb = LearnerRegrXgboost$new()
  xgbps <- learner_xgb$param_set$clone(deep = TRUE)
  learner_xgb$param_set$values$booster <- "gblinear"
  learner_xgb$param_set$values$eval_metric <- "error"
  learner_xgb$param_set$values$nrounds <- "200"
  
  xgbps$subset(c("eta", "max_depth", "alpha", "lambda"))
  xgbps$params[1]$eta$upper <- 0.4
  xgbps$params[2]$max_depth$upper <- 30
  xgbps$params[3]$alpha$upper <- 0.4
  xgbps$params[4]$lambda$upper <- 2
  
  xgbtuner = tnr("random_search")
  xgbevals <- term("evals", n_evals = 50)
  
  xgbat = AutoTuner$new(
    learner = learner_xgb,
    resampling = rsmp_cv,
    measures = measure,
    tune_ps = xgbps,
    terminator = xgbevals,
    tuner = xgbtuner
  )
  xgbat$param_set$trafo = transform
  xgbat$train(task)
  
  # decision tree
  learner_rpart = LearnerRegrRpart$new()
  dtps <- learner_rpart$param_set$clone(deep = TRUE)
  
  dttuner = tnr("random_search")
  dtevals <- term("evals", n_evals = 1)
  
  learner_rpart$param_set$trafo = transform
  learner_rpart$train(task)
  
  # we can save the predictions as data.table and export them for Kaggle
  test_set = read.csv("data/ames_housing_test_numeric.csv")
  pred = learner_rpart$predict_newdata(task, test_set)
  pred = as.data.table(pred)
  pred$truth = NULL
  write.csv(pred, "data/ames_housing_submission_rpart.csv", row.names = FALSE)
