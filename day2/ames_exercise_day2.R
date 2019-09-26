# load packages
library(mlr3)
library(mlr3learners)
library(mlr3tuning)
library(magrittr)

train_set = read.csv("data/ames_housing_train_numeric.csv")

# the task
task = TaskRegr$new(id = "ames_housing", backend = train_set, target = "SalePrice")

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
evals <- term("evals", n_evals = 20)

learner_ranger$param_set$values$splitrule <- "variance"
learner_ranger$param_set$values$importance <- "permutation"
learner_ranger$param_set$values$scale.permutation.importance <- FALSE 

ps <- learner_ranger$param_set$clone(deep = TRUE)
ps$subset(c("num.trees", "min.node.size"))
ps$params[1]$num.trees$upper <- 250
ps$params[2]$min.node.size$upper <- 30

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
write.csv(pred, "data/ames_housing_submission.csv", row.names = FALSE)

ggplot(at$archive(unnest = "params"),
       aes(x = min.node.size, y = regr.mae, color = scale.permutation.importance)) + geom_point()


# lets do a knn version
learner_knn = lrn("regr.kknn")
kps <- learner_knn$param_set$clone(deep = TRUE)
ps$subset(c("k", "distance", "kernel"))
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
kat$param_set$trafo = transform
kat$train(task)

ggplot(kat$archive(unnest = "params"),
       aes(x = distance, y = regr.mae, color = scale)) + geom_point()
