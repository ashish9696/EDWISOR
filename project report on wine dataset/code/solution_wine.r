rm(list=ls())
library(corrplot) 
library(caret)    
library(randomForest)  
library(gridExtra) 
library(MASS)
library(doSNOW)    
library(car)
registerDoSNOW(makeCluster(4, type = 'SOCK'))

#set today's date
today <- as.character(Sys.Date())

#set working directory
setwd("H:/edwisor/CaseStudyOnWines/")

#load datasets
white <- read.csv('winequality-white.csv', sep=';')
red <- read.csv('winequality-red.csv', sep=';')

#label dataset
red[, 'color'] <- 'red'
white[, 'color'] <- 'white'

#rowbind datasets
df <- rbind(red, white)
attach(df)
summary(df$alcohol)

#exploratory data analysis
par(mfrow=c(2,2), oma = c(1,1,0,0) + 0.1, mar = c(3,3,1,1) + 0.1)
barplot((table(quality)), col=c("slateblue4", "slategray", "slategray1", "slategray2", "slategray3", "skyblue4"))
mtext("Quality", side=1, outer=F, line=2, cex=0.8)
truehist(fixed.acidity, h = 0.5, col="slategray3")
mtext("Fixed Acidity", side=1, outer=F, line=2, cex=0.8)
truehist(volatile.acidity, h = 0.05, col="slategray3")
mtext("Volatile Acidity", side=1, outer=F, line=2, cex=0.8)
truehist(citric.acid, h = 0.1, col="slategray3")
mtext("Citric Acid", side=1, outer=F, line=2, cex=0.8)
par(mfrow=c(1,5), oma = c(1,1,0,0) + 0.1,  mar = c(3,3,1,1) + 0.1)
boxplot(fixed.acidity, col="slategray2", pch=19)
mtext("Fixed Acidity", cex=0.8, side=1, line=2)
boxplot(volatile.acidity, col="slategray2", pch=19)
mtext("Volatile Acidity", cex=0.8, side=1, line=2)
boxplot(citric.acid, col="slategray2", pch=19)
mtext("Citric Acid", cex=0.8, side=1, line=2)
boxplot(residual.sugar, col="slategray2", pch=19)
mtext("Residual Sugar", cex=0.8, side=1, line=2)
boxplot(chlorides, col="slategray2", pch=19)
mtext("Chlorides", cex=0.8, side=1, line=2)
###

##summary and correlation
summary=summary(df)
library("psych")
describe(df)
cor(df[,-13])
cor(df[,-13], method="spearman")
pairs(df[,-13], gap=0, pch=19, cex=0.4, col="darkblue")
title(sub="Scatterplot of Chemical Attributes", cex=0.8)
###

##data preparation
limout <- rep(0,11)
for (i in 1:11){
  t1 <- quantile(df[,i], 0.75)
  t2 <- IQR(df[,i], 0.75)
  limout[i] <- t1 + 1.5*t2
}
dfIndex <- matrix(0, 4898, 11)
for (i in 1:4898)
  for (j in 1:11){
    if (df[i,j] > limout[j]) dfIndex[i,j] <- 1
  }
WWInd <- apply(dfIndex, 1, sum)
dfTemp <- cbind(WWInd, df)
Indexes <- rep(0, 208)
j <- 1
for (i in 1:4898){
  if (WWInd[i] > 0) {Indexes[j]<- i
  j <- j + 1}
  else j <- j
}
df <-df[-Indexes,]   # Inside of Q3+1.5IQR
indexes = sample(1:nrow(df), size=0.5*nrow(df))
train <- df[indexes,]
indexes <- df[-indexes,]
###
#make excel of dataset
library(xlsx)
write.xlsx(df, "H:/edwisor/CaseStudyOnWines/df.xlsx")

## analysis

df$color <- as.factor(df$color)
good_ones <- df$quality >= 5
#below_good <- (df$quality >=7 & df$quality < 9)
#normal <- (df$quality >=5 & df$quality < 7)
#average <- (df$quality >=3 & df$quality < 5)
poor <- df$quality <5
df[good_ones, 'quality'] <- 'best'
#df[below_good, 'quality'] <- 'better'
#df[normal, 'quality'] <- 'good'
#df[average, 'quality'] <- 'average'
df[poor, 'quality'] <- 'poor'
df$quality <- as.factor(df$quality)
dummies <- dummyVars(quality ~ ., data = df)
df_dummied <- data.frame(predict(dummies, newdata = df))
df_dummied[, 'quality'] <- df$quality

# set the seed for reproducibility
set.seed(1234) 
trainIndices <- createDataPartition(df_dummied$quality, p = 0.7, list = FALSE)
train <- df_dummied[trainIndices, ]
test <- df_dummied[-trainIndices, ]

#1
numericColumns <- !colnames(train) %in% c('quality', 'color.red', 'color.white')
correlationMatrix <- cor(train[, numericColumns])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff = 0.6)
colnames(correlationMatrix)[highlyCorrelated]

png(paste0(today, '-', 'correlation-matrix.png'))
corrplot(correlationMatrix, method = 'number', tl.cex = 0.5)
dev.off()

#2
fitControl_rfe <- rfeControl(functions = rfFuncs, method = 'cv', number = 5) # 5-fold CV
fit_rfe <- rfe(quality ~., data = train,
               sizes = c(1:10),  # subset sizes to test (ahem... not sure how it works)
               rfeControl = fitControl_rfe)
features <- predictors(fit_rfe) # same command as fit_rfe$optVariables
max(fit_rfe$results$Accuracy)

png(paste0(today, '-', 'recursive-feature-elimination.png'))
plot(fit_rfe, type = c('g', 'o'), main = 'Recursive Feature Elimination')
dev.off()

# Normalize the quantitative variables to be within the [0,1] range
train_normalized <- preProcess(train[, numericColumns], method = 'range')
train_plot <- predict(train_normalized, train[, numericColumns])

# Let's take an initial peek at how the predictors separate on the target
png(paste0(today, '-', 'feature-plot.png'))
featurePlot(train_plot, train$quality, 'box')
dev.off()

# fitControl <- trainControl(method = 'repeatedcv', number = 5, repeats = 3)
fitControl <- trainControl(method = 'repeatedcv', number = 5, repeats = 3)

#rf
# tunable parameter: mtry (number of variables randomly sampled as candidates at each split)
fit_rf <- train(x = train[, -14], y = train$quality,
                method = 'rf',
                # preProcess = 'range', # it seems slightly better without 'range'
                trControl = fitControl,
                tuneGrid = expand.grid(.mtry = c(2:6)),
                n.tree = 1000) 
predict_rf <- predict(fit_rf, newdata = test[, 0:14])
confusionMatrix(predict_rf, test$quality, positive = 'best')
confMat_rf <- confusionMatrix(predict_rf, test$quality, positive = 'best')
importance_rf <- varImp(fit_rf, scale = TRUE)

png(paste0(today, '-', 'importance-rf.png'))
plot(importance_rf, main = 'Feature importance for Random Forest')
dev.off()

