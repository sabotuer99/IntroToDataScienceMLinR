require(neuralnet)

setwd("/home/troywhorten/intro_ds/day2")

transform <- function(x) { 
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

train_raw <- read.csv("data/ames_housing_train_numeric.csv")
test_raw <- read.csv("data/ames_housing_test_numeric.csv")

train_scale <- scale(transform(train_raw))
test_scale <- scale(transform(test_raw), scale = attr(train_scale, 'scaled:scale')[-34], center = attr(train_scale, 'scaled:center')[-34])

train_set <- as.data.frame(train_scale)
test_set <- as.data.frame(test_scale)

unscale <- function (col){
  col * attr(train_scale, 'scaled:scale')["SalePrice"] + attr(train_scale, 'scaled:center')["SalePrice"]
}

n <- names(train_set)
f <- as.formula(paste("SalePrice ~", paste(n[!n %in% "SalePrice"], collapse = " + ")))
nn <- neuralnet(f,data=train_set,hidden=c(16, 8, 4),linear.output=T, threshold = 0.02, stepmax = 1e+06)

plot(nn)

pr.nn <- compute(nn,test_set)
response <- unscale(pr.nn$net.result)
row_id <- seq.int(nrow(response)) + 1953
result <- as.data.table(cbind(row_id, response))
colnames(result) <- c("row_id", "response")

write.csv(result, "data/ames_housing_submission_nn.csv", row.names = FALSE)

