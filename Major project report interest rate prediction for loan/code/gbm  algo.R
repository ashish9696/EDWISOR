rm(list=ls())
library(corrplot) 
library(caret)    
library(randomForest)  
library(gridExtra) 
library(MASS)
library(doSNOW)    
library(car)
library(mlbench)
library(h2o)
library(stringr)
library(tm)
library(wordcloud)
library(slam)
library(sentimentr)
registerDoSNOW(makeCluster(3, type = 'SOCK'))
library(data.table)
#set today's date
today <- as.character(Sys.Date())
#set working directory
path <- "H:/project/"
setwd(path)
getwd()
data <- read.csv("data_preprocessed.csv", stringsAsFactors = T)
data$X28=NULL
set.seed(1234)
library(caret) 
trainIndices <- createDataPartition(data$X1, p = 0.8, list = FALSE)
train <- data[trainIndices, ]
test <- data[-trainIndices, ]
library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)
colnames(train.h2o)
y.dep <- 1
x.indep <- c(2:27)
#GBM
system.time(
  gbm.model <- h2o.gbm(y=y.dep, x=x.indep, training_frame = train.h2o, ntrees = 1000, max_depth = 4, learn_rate = 0.01, seed = 1122)
)
##performance check
h2o.performance(gbm.model)
predict.dl2 <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbmlearning <- data.frame(X1 = test$X1,X2 = test$X2,X3 = test$X3, X1_predict =  predict.dl2$predict)
write.csv(sub_gbmlearning, file = "sub_gbm_final_result.csv", row.names = F)
#predict on test cases


#predict
predict_data <- read.csv('Holdout for Testing.csv', stringsAsFactors = T)

##preprocess
predict_data$X12 = factor(predict_data$X12, labels=(1:length(levels(factor(predict_data$X12)))))
predict_data$X12 = as.numeric(predict_data$X12)
predict_data$X32 = factor(predict_data$X32, labels=(1:length(levels(factor(predict_data$X32)))))
predict_data$X32 = as.numeric(predict_data$X32)
predict_data$X14 = factor(predict_data$X14, labels=(1:length(levels(factor(predict_data$X14)))))
predict_data$X14 = as.numeric(predict_data$X14)
predict_data$X17 = factor(predict_data$X17, labels=(1:length(levels(factor(predict_data$X17)))))
predict_data$X17 = as.numeric(predict_data$X17)
predict_data$X20 = factor(predict_data$X20, labels=(1:length(levels(factor(predict_data$X20)))))
predict_data$X20 = as.numeric(predict_data$X20)
predict_data$X8 = factor(predict_data$X8, labels=(1:length(levels(factor(predict_data$X8)))))
predict_data$X8 = as.numeric(predict_data$X8)
predict_data$X9 = factor(predict_data$X9, labels=(1:length(levels(factor(predict_data$X9)))))
predict_data$X9 = as.numeric(predict_data$X9)

#outlier analysis
predict_data$X4 <- as.numeric(predict_data$X4)
predict_data$X5 <- as.numeric(predict_data$X5)
predict_data$X6 <- as.numeric(predict_data$X6)
predict_data$X6 <- as.numeric(predict_data$X7)

predict_data$X8 <- as.numeric(predict_data$X8)
predict_data$X9 <- as.numeric(predict_data$X9)
predict_data$X12 <- as.numeric(predict_data$X12)

predict_data$X27 <- as.numeric(predict_data$X27)
predict_data$X28 <- as.numeric(predict_data$X28)
predict_data$X7<-gsub("months", " ", predict_data$X7)
predict_data$X1<-gsub("%", " ", predict_data$X1)
predict_data$X19<-gsub("xx", " ", predict_data$X19)
predict_data$X30<-gsub("%", " ", predict_data$X30)

write.csv(predict_data, file = "predict_data.csv", row.names = F)
predict_data <- read.csv('predict_data.csv', stringsAsFactors = T)
test.h2o <- as.h2o(predict_data)
predict.gbm <- as.data.frame(h2o.predict(gbm.model, test.h2o))
sub_gbm <- data.frame(X1 = predict_data$X1,X2 = predict_data$X2,X3 = predict_data$X3, X1_predict =  predict.gbm$predict)
write.csv(sub_gbm, file = "sub_gbm_final result.csv", row.names = F)


