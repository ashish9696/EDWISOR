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
train <- read.csv("Data for Cleaning & Modeling.csv", stringsAsFactors = T)
test <- fread('Holdout for Testing.csv', stringsAsFactors = T)
train<-data.table(train)
meta <- read.csv('Metadata.csv', sep=',')
test$X1=NULL
#combine data set
train$X1<-gsub("%", " ", train$X1)
test[,X1 := mean(train$X1)]
c <- list(train, test)
data <- rbindlist(c)

View(data)
str(data)
table(data$X1)
head(data)
tail(data)
attach(data)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
##word cloud
#Wordcloud for negative terms
negCorpus = Corpus(VectorSource(data$X16))
negCorpus = tm_map(negCorpus, tolower)
negCorpus = tm_map(negCorpus, removeWords, c('will','can','pay','borrower','added', stopwords('english')))
negCorpus = tm_map(negCorpus, removePunctuation)
negCorpus = tm_map(negCorpus, removeNumbers)
negCorpus = tm_map(negCorpus, stripWhitespace)
negCorpus = tm_map(negCorpus, PlainTextDocument)
negCorpus = Corpus(VectorSource(negCorpus))

#wordcloud(negCorpus, max.words = 200, scale=c(3, .1), colors=brewer.pal(6, "Dark2"))

# wordcloud
pal2 = brewer.pal(8,"Dark2")
png("wordcloud_reason1.png", width=12,height=8, units='in', res=300)
wordcloud(negCorpus, scale=c(5,.2),min.freq=20, max.words=150, random.order=FALSE, rot.per=.15, colors=pal2)
dev.off()
##

names(which(sapply(data, function(x)sum(is.na(x))>=1)=="TRUE"))
class(data)
sapply(data, function(x) length(unique(x)))
#Count of no of na's values in a column
sapply(data, function(x) sum(is.na(x)))
apply(data, 2, function(x)sum(is.na(x)))
#Store Values in data frame
MissingData = data.frame(varaibles = colnames(data), MissingInfo = apply(data,2,function(x)sum(is.na(x))))
#Store UNIQUE Values in data frame
UNIQUEData = data.frame(varaibles = colnames(data), sapply(data, function(x) length(unique(x))))
#leaVE variable more na's
data$X26=NULL
data$X25=NULL
data$X16=NULL
data$X10=NULL
data$X18=NULL
data$X13[is.na(data$X13)] = median(data$X13, na.rm = TRUE) #0
## feature engineering
#Sorting in ascending order
data$X7<-gsub("months", " ", data$X7)
data$X1<-gsub("%", " ", data$X1)
data$X19<-gsub("xx", " ", data$X19)
data$X30<-gsub("%", " ", data$X30)
data = data[-1,]
data$X12 = factor(data$X12, labels=(1:length(levels(factor(data$X12)))))
data$X12 = as.numeric(data$X12)
data$X32 = factor(data$X32, labels=(1:length(levels(factor(data$X32)))))
data$X32 = as.numeric(data$X32)
data$X14 = factor(data$X14, labels=(1:length(levels(factor(data$X14)))))
data$X14 = as.numeric(data$X14)
data$X17 = factor(data$X17, labels=(1:length(levels(factor(data$X17)))))
data$X17 = as.numeric(data$X17)
data$X20 = factor(data$X20, labels=(1:length(levels(factor(data$X20)))))
data$X20 = as.numeric(data$X20)
data$X8 = factor(data$X8, labels=(1:length(levels(factor(data$X8)))))
data$X8 = as.numeric(data$X8)
data$X9 = factor(data$X9, labels=(1:length(levels(factor(data$X9)))))
data$X9 = as.numeric(data$X9)

data = data[order(data$X8, data$X9 , data$X7, data$X12, data$X32, data$X14 , data$X17, data$X8,data$X9),]

#remove na's
data = data[complete.cases(data),]
#outlier analysis
library(outliers)
data$X4 <- as.numeric(data$X4)
data$X5 <- as.numeric(data$X5)
data$X6 <- as.numeric(data$X6)
data$X27 <- as.numeric(data$X27)
data$X28 <- as.numeric(data$X28)
##find
boxplot(data$X4, col="slategray2", pch=19)
boxplot(data$X5, col="slategray2", pch=19)
boxplot(data$X6, col="slategray2", pch=19)

boxplot(data$X13, col="slategray2", pch=19)
boxplot(data$X21, col="slategray2", pch=19)
boxplot(data$X22, col="slategray2", pch=19)
boxplot(data$X24, col="slategray2", pch=19)
boxplot(data$X27, col="slategray2", pch=19)
boxplot(data$X28, col="slategray2", pch=19)
boxplot(data$X29, col="slategray2", pch=19)
boxplot(data$X31, col="slategray2", pch=19)
##summary and correlation
summary=summary(data)
##data preparation
##outlier for x21

for (i in 1:49) {
  
  
  outlier_tf = outlier(data$X21, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
  
}
boxplot(data$X21, col="slategray2", pch=19)
##outlier for x13
for (i in 1:6) {
  
  
  outlier_tf = outlier(data$X24, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
}
boxplot(data$X24, col="slategray2", pch=19)
##outliers for some
for(i in 1:36)
{
  
  
  outlier_tf = outlier(data$X27, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
}
boxplot(data$X27, col="slategray2", pch=19)
##outliers for some
for(i in 1:21)
{
  
  
  outlier_tf = outlier(data$X28, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
}
boxplot(data$X28, col="slategray2", pch=19)
# some outliers

#outlier analysis
for(i in 1:48)
{
  
  
  outlier_tf = outlier(data$X31, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
}
boxplot(data$X31, col="slategray2", pch=19)
#outlier some
for(i in 1:4500)
{
  
  
  outlier_tf = outlier(data$X13, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
  
  
  }
boxplot(data$X13, col="slategray2", pch=19)
##outliers more
for(i in 1:12000)#14000
{
  outlier_tf = outlier(data$X29, logical = T)
  find_outlier = which(outlier_tf == TRUE, arr.ind = TRUE)
  data = data[-find_outlier,]
}
boxplot(data$X29, col="slategray2", pch=19)

boxplot(data$X1, col="slategray2", pch=19)
data$X1 = as.numeric(data$X1)

data$X1[is.na(data$X1)] = median(data$X1, na.rm = TRUE)
## outliers removed
##now unique values
#Count of unique values in a column
sapply(data, function(x) length(unique(x)))
#Count of no of na's values in a column
sapply(data, function(x) sum(is.na(x)))
library(xlsx)
write.csv(data, file = "data_preprocessed.csv", row.names = F)

data$X28=NULL
data$X7=NULL
data$X19=NULL
data$X30=NULL
## sampling
# set the seed for reproducibility
set.seed(1234)
library(caret) 
trainIndices <- createDataPartition(data$X1, p = 0.8, list = FALSE)
train <- data[trainIndices, ]
test <- data[-trainIndices, ]
#devide test in more
# h2o.rm(object= localH2O, keys= "prostate.train")
library(h2o)
localH2O <- h2o.init(nthreads = -1)
h2o.init()
train.h2o <- as.h2o(train)
test.h2o <- as.h2o(test)
colnames(train.h2o)
y.dep <- 1
x.indep <- c(2:23)
#deep learning
system.time(
  dlearning.model <- h2o.deeplearning(y = y.dep,
                                      x = x.indep,
                                      training_frame = train.h2o,
                                      epoch = 60,
                                      hidden = c(100,100),
                                      activation = "Rectifier"
  )
)
##performance check
h2o.performance(dlearning.model)
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))
sub_dlearning <- data.frame(X1 = test$X1,X2 = test$X2,X3 = test$X3, X1_predict =  predict.dl2$predict)
importance(dlearning.model, type = 1)
write.csv(sub_dlearning, file = "sub_dlearning_accuracy.csv", row.names = F)
#predict on test cases
pred = h2o.predict(dlearning.model, test.h2o)
pred_prob = h2o.predict(dlearning.model,test.h2o, type = "raw")



#predict
predict_data <- read.csv('Holdout for Testing.csv', sep=',')
predict_data$X4=as.numeric(predict_data$X4)
predict_data$X5=as.numeric(predict_data$X5)
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
predict_data$X27 <- as.numeric(predict_data$X27)
predict_data$X28 <- as.numeric(predict_data$X28)
predict_data$X7<-gsub("months", " ", predict_data$X7)
predict_data$X1<-gsub("%", " ", predict_data$X1)
predict_data$X19<-gsub("xx", " ", predict_data$X19)
predict_data$X30<-gsub("%", " ", predict_data$X30)

test.h2o <- as.h2o(predict_data)
predict.dl2 <- as.data.frame(h2o.predict(dlearning.model, test.h2o))
sub_dlearning <- data.frame(X1 = predict_data$X1,X2 = predict_data$X2,X3 = predict_data$X3, X1_predict =  predict.dl2$predict)
write.csv(sub_dlearning, file = "sub_dlearning_final result.csv", row.names = F)
