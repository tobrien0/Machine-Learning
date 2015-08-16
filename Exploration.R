rm(list=ls())

### Load libraries and set variables
### ----------------------------------------------------------------------------
library(downloader);
library(lattice); library(ggplot2);
suppressWarnings(library(caret)); library(caret);
suppressWarnings(library(rpart)); library(rpart);
suppressWarnings(library(randomForest)); library(randomForest);
# library(knitr);
suppressWarnings(library(rattle)); library(rattle);
suppressWarnings(library(rpart.plot)); library(rpart.plot);
#library(doParallel)

dirData <- "./data"
if(!file.exists(dirData)){dir.create(dirData)}
fileTrain <- paste(dirData, "pml-training.csv", sep="/")
fileTest <- paste(dirData, "pml-testing.csv", sep="/")
urlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

### Load Data
### ----------------------------------------------------------------------------
### Download the src data
if(!file.exists(fileTrain)){download(urlTrain, dest=fileTrain, mode="wb")}
if(!file.exists(fileTest)){download(urlTest, dest=fileTest, mode="wb")}

### "record" date downloaded
dateDownloaded <- date()
dateDownloaded

### Load the raw data, indicate NA values and close connections
dfTrain <- read.csv(fileTrain, na.strings=c("NA", "DIV/0", ""))
dfTest <- read.csv(fileTest, na.strings=c("NA", "DIV/0", ""))
closeAllConnections()

### Preliminary Exploration
### ----------------------------------------------------------------------------
dim(dfTrain)
dim(dfTest)

### Clean and transform data
### ----------------------------------------------------------------------------
### remove columns not relevant for analysis
dfTrain <- dfTrain[, -c(1:7)]
dfTest <- dfTest[, -c(1:7)]

### Prune columns with zero values (using nzv only reduces columns to 52 instead of 53)
dfTrain <- dfTrain[, colSums(is.na(dfTrain)) == 0]
dfTest <- dfTest[, colSums(is.na(dfTest)) == 0]

dim(dfTrain)
dim(dfTest)

# 53 is a still lot of predictors and it takes a long time to train the models,
# so I will perform some correlation analysis to see if additional variables can be
# reduced.
correlation <- cor(dfTrain[sapply(dfTrain, is.numeric)])
dim(correlation)
removeCols <- findCorrelation(correlation, cutoff=0.90, verbose=TRUE)
dfTrain <- dfTrain[, -removeCols]
dfTest <- dfTest[, -removeCols]
dim(dfTrain)
dim(dfTest)
# Down to a manageable number of predictors (46)

### Transform class into factor
dfTrain$classe <- factor(dfTrain$classe)

### Partition the training set to allow for cross-validation
### ----------------------------------------------------------------------------
set.seed(1234)
dfPartition <- createDataPartition(y=dfTrain$classe, p=0.6, list=FALSE)
training <- dfTrain[dfPartition, ]
testing <- dfTrain[-dfPartition, ]

### Additional Exploration
dim(training)
dim(testing)
unique(training$classe)

#qplot(classe,colour=classe, data=training)

descMain <- "Bar Plot of Frequency by Classe\nfor the Training Partition of the Training Set (60%)"
plot(training$classe,
     col="green",
     main=descMain,
     xlab="Classe",
     ylab="Frequency",
     ylim=c(0, 3500)
)

# The partitioned data sets have a consistent number of variables with 60% of the
# observations allocated to training and 40% to testing. There are five unique
# values in the classe variable (A-E) with frequencies ranging from a low of just
# less than 2000 for D to a high of ~3300 for A.

#registerDoParallel(makeCluster(detectCores()))
#names(getModelInfo())

### Generate models
### ----------------------------------------------------------------------------
### CART - model 1
model1 <- train(classe ~ ., data=training, method="rpart")
print(model1$finalModel)
fancyRpartPlot(model1$finalModel)
prediction1 <- predict(model1, testing)
confusionMatrix(prediction1, testing$classe)
# Cross-validation accuracy of 49.25% and confidence interval of 48.14% isn't very good

### Boosted trees - model 2
tuneGrid <- expand.grid(n.trees=seq(1,501,10),
                        interaction.depth=2:5,
                        shrinkage=0.1,
                        n.minobsinnode=10)
fitControl <- trainControl(method="repeatedcv",
                           number=3,
                           repeats=1,
                           verboseIter=FALSE,
                           returnResamp="all")
model2 <- train(classe ~ .,
                data=training,
                method="gbm",
                trControl=fitControl,
                tuneGrid=tuneGrid)

prediction2<- predict(model2, testing)
confusionMatrix(prediction2, testing$classe)
### Cross-validation accuracy of 99.22% and the CI is at least 99.0. Not quite as good
### as random forest and it took a long time to run

### Linear discriminant analysis - Model 3
model3 <-train(classe ~ ., data=training, method='lda')
prediction3 <- predict(model3, testing)
confusionMatrix(prediction3, testing$classe)
### Cross-validation accuracy of 67.51% isn't very good

### Random forest - Model 4
model4 <- train(classe ~ ., data=training, method="rf")
print(model4$finalModel)
prediction4 <- predict(model4, testing)
cm4 <- confusionMatrix(prediction4, testing$classe)
round(cm4$overall['Accuracy']*100, 2)
round(cm4$overall['AccuracyLower']*100, 2)

varImp(model4)
varImptPlot(model4, top=20, main="Top 20 Variables by Importance")
### Cross-validation accuracy of 99.07% and the CI is at least 98.53

### if caret training is too slow, try this...
model4a <- randomForest(classe~., data=training, importance=TRUE, keep.forest=TRUE)
prediction4a <- predict(model4a, testing)
cm4a <- confusionMatrix(prediction4a, testing$classe)
str(cm4a)
print(cm4a)
round(cm4a$overall['Accuracy']*100, 2)
round(cm4a$overall['AccuracyLower']*100, 2)

importance(model4a)
varImpPlot(model4a, main="Top 20 Variables by Importance", scale=FALSE, top=20, class="Yes")
varImpPlot(model4a, main="Top 20 Variables by Importance")



# fewer predictors runs faster but not very accurate
model2b <- randomForest(classe ~ pitch_forearm +
                                magnet_belt_y +
                                magnet_dumbbell_y +
                                roll_forearm +
                                accel_forearm_x,
                        data=training,
                        importance=TRUE,
                        keep.forest=TRUE)
prediction2b <- predict(model2b, testing)
cm2b <- confusionMatrix(prediction2b, testing$classe)
cm2b


### Prediction and output files
### ----------------------------------------------------------------------------
# predict outcome on the testing data set using Random Forest algorithm
predictedOutcome <- predict(model4a, dfTest)
predictedOutcome

writeOutput=function(x){
        n=length(x)
        for(i in 1:n){
                filename=paste0("output_",i,".txt")
                write.table(x[i],
                            file=filename,
                            quote=FALSE,
                            row.names=FALSE,
                            col.names=FALSE)
        }
}

writeOutput(predictedOutcome)
