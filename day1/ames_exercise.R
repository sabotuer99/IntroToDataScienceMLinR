# load packages
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
  

# load the data DONT NEED USED RSTUDIO DATASET IMPORT
train_set = read.csv("data/ames_housing_train.csv")
train_set$PID <- 1L

# the task
task = TaskRegr$new(id = "ames_housing", backend = train_set, target = "SalePrice")

#transformation function
transform <- function(x, param_set) { 
  x$Bedroom.AbvGr <- factor(x$Bedroom.AbvGr)
  x 
}

# create a learner
#learner_featureless = lrn("regr.featureless")
learner_ranger <- LearnerRegrRanger$new()
learner_xgboost <- lrn("regr.xgboost", booster = "gbtree", eval_metric = "mse")
learner_svm = lrn("regr.svm")
learner_knn = lrn("regr.kknn", k = 15)

learner_knn$param_set$trafo <- transform
learner_xgboost$param_set$trafo <- transform
learner_ranger$param_set$trafo <- transform


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
ps <- learner_xgboost$param_set$clone(deep = TRUE)
#ps <- learner_ranger$param_set$clone(deep = TRUE)
#ps$subset(c("num.trees", "min.node.size", "importance", "splitrule", "scale.permutation.importance"))
#ps$params[1]$num.trees$upper <- 1000
#ps$params[2]$min.node.size$upper <- 1000
#ps$params[3]$importance$default <- "none"

instance = TuningInstance$new(
  task = task,
  learner = learner_ranger,
  resampling = hout,
  measures = measure,
  param_set = ps,
  terminator = evals20
)
tuner = tnr("grid_search", resolution = 5)
result = tuner$tune(instance)

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
                             learner_knn))
                             # lrn("regr.glmnet", predict_type = "response"),
                             # lrn("regr.rpart", predict_type = "response"),
                             # lrn("regr.ranger", predict_type = "response"),
                             # lrn("regr.kknn", predict_type = "response"),
                             # lrn("regr.xgboost", predict_type = "response"),
                             # lrn("regr.featureless", predict_type = "response")))
bmr = benchmark(bm_design)
performances = bmr$aggregate(list(msr("regr.mse")))
performances[, c("learner_id", "regr.mse")]


test_set = read.csv("data/ames_housing_test.csv")
pred = learner_ranger$predict_newdata(task, test_set)

# we can save the predictions as data.table and export them for Kaggle
pred = as.data.table(pred)
pred$truth = NULL
write.csv(pred, "data/ames_housing_submission.csv", row.names = FALSE)