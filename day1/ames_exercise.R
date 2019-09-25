# load packages
library(mlr3)
library(mlr3learners)

# load the data
train_set = read.csv("data/ames_housing_train.csv")

# the task
task = TaskRegr$new(id = "ames_housing", backend = train_set, target = "SalePrice")

# create a learner
learner_featureless = lrn("regr.featureless")
learner_ranger = lrn("classif.log_reg")
learner_cart = lrn("classif.rpart", maxdepth = 5)
learner_knn = lrn("classif.kknn")

# train the learner
# EXCHANGE THIS PART WITH A PROPER BENCHMARK ANALYSIS!
learner_featureless$train(task)
learner_ranger$train(task)
learner_cart$train(task)
learner_knn$train(task)

# predict on new data
# CHOOSE THE BEST PERFOMING LEARNER OF YOUR BENCHMARK TO PREDICT ON NEW DATA!
test_set = read.csv("data/ames_housing_test.csv")
pred = learner$predict_newdata(task, test_set)

# we can save the predictions as data.table and export them for Kaggle
pred = as.data.table(pred)
pred$truth = NULL
write.csv(pred, "data/ames_housing_submission.csv", row.names = FALSE)