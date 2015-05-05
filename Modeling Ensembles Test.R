library(caret)
library(pROC)
library(RCurl)

#Downloading test data set
urlfile <- 'https://raw.githubusercontent.com/hadley/fueleconomy/master/data-raw/vehicles.csv'
x <- getURL(urlfile, ssl.verifypeer = FALSE)
vehicles <- read.csv(textConnection(x))

#Cleaning cyclinder data to identify 6 cylinder vehicles
vehicles <- vehicles[names(vehicles)[1:24]]
vehicles <- data.frame(lapply(vehicles, as.character), stringsAsFactors = FALSE)
vehicles <- data.frame(lapply(vehicles, as.numeric))
vehicles[is.na(vehicles)] <- 0
vehicles$cylinders <- ifelse(vehicles$cylinder == 6, 1, 0)

#Instead of splitting into two sets like usual, we need three for ensembles
set.seed(1234)
vehicles <- vehicles[sample(nrow(vehicles)),]
split <- floor(nrow(vehicles)/3)
ensembleData <- vehicles[0:split,]
blenderData <- vehicles[(split+1):(split*2),]
testingData <- vehicles[(split*2+1):nrow(vehicles),]

#Labeling outcome and predictor names
labelName <- "cylinders"
predictors <- names(ensembleData)[names(ensembleData) != labelName]

#Using caret to control the number of CV's performed
myControl <- trainControl(method = 'cv', number = 3, returnResamp = 'none')

#Benchmark model
#Set up to use as a comparative benchmark for our ensembling models
test_model <- train(blenderData[, predictors], blenderData[, labelName],
                    method = 'gbm', trControl = myControl)

#Using benchmark model to predict 6 cylinder vehicles using testingData
preds <- predict(object = test_model, testingData[, predictors])
auc <- roc(testingData[, labelName], preds)
print(auc$auc)

#Ensembles
#Train 3 different models
model_gbm <- train(ensembleData[, predictors], ensembleData[,labelName], 
                   method = 'gbm', trControl = myControl)
model_rpart <- train(ensembleData[, predictors], ensembleData[, labelName],
                     method = 'rpart', trControl = myControl)
model_treebag <- train(ensembleData[, predictors], ensembleData[, labelName],
                       method = 'treebag', trControl = myControl)


