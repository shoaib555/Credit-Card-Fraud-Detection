rm(list=ls())
library(data.table)
cc=fread("CC.csv",sep=",")
summary(cc)
dim(cc)
str(cc)
library(DataExplorer)
plot_histogram(cc)
plot_density(cc)
sapply(cc, function(x) sum(is.na(x)))
cc=cc[,-1]
library(corrplot)
corrplot(cor(cc[,-c(30)]),type="upper",method="number")


cc1=scale(cc[,-30])
cc=cbind(cc1,cc$Class)
cc=as.data.frame(cc)
colnames(cc)[30]="Class"
cc$Class=as.factor(cc$Class)

prop.table(table(cc$Class))
library(DMwR)
ccs=SMOTE(cc$Class~.,data=cc,perc.over = 300)
prop.table(table(ccs$Class))
ccs=as.data.frame(ccs)
set.seed(234)
library(caTools)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)

set.seed(123)
lg=glm(tr$Class~.,data=tr,family = binomial(link="logit"))
summary(lg)
ts$pred=predict(lg,ts,type="response")
ts$prob=ifelse(ts$pred>0.3,1,0)
ts$prob=as.factor(ts$prob)
table(ts$prob)
table(ts$Class)

library(caret)
confusionMatrix(ts$prob,ts$Class,positive = "1")

library(ROCR)
ROCRpred = prediction(ts$pred, ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.3),text.adj=c(-0.5,1.7),main="ROC for Logistic")



library(rpart)
library(rpart.plot)
library(rattle)
set.seed(234)
library(caTools)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)


set.seed(897)


r.ctrl=rpart.control(minisplit=50,minbucket=10,cp=0,xval=10)
dt=rpart(formula=Class~.,data=tr,control = r.ctrl)
plotcp(dt)

r.ctrl=rpart.control(minisplit=50,minbucket=10,cp=0.0031,xval=10)
dt1=rpart(formula=Class~.,data=tr,control = r.ctrl)
ts$pred=predict(dt1,ts,type="prob")[,"1"]
ts$prob=ifelse(ts$pred>0.5,1,0)
ts$prob=as.factor(ts$prob)

library(caret)
confusionMatrix(ts$prob,ts$Class,positive = "1")

fancyRpartPlot(dt1)

library(ROCR)
ROCRpred = prediction(ts$pred, ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.5,1.7),main="ROC for CART")

library(randomForest)
#set seed again for randomness
set.seed(1000)
library(caTools)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)
#build first RF model
set.seed(555)
rf=randomForest(Class~.,data=tr,ntree=200,mtry=3,nodesize=10,importance=T)
print(rf)
plot(rf)

#tune rf to identify the best mtry
set.seed(1000)
trrf=tuneRF(tr[,-c(30)],y=tr$Class,mtryStart = 3,stepFactor = 1.5,ntree=100,improve = 0.0001,nodesize=10,
            trace=T,plot=T,doBest = T,importance=T)
plot(trrf)
print(trrf)

rf=randomForest(Class~.,data=tr,ntree=100,mtry=6,nodesize=10,importance=T)
print(rf)
plot(rf)

ts$pred=predict(rf,ts,type="prob")[,"1"]
ts$prob=ifelse(ts$pred>0.2,1,0)
ts$prob=as.factor(ts$prob)
table(ts$prob)
table(ts$Class)

library(caret)
confusionMatrix(ts$prob,ts$Class,positive = "1")

library(ROCR)
ROCRpred = prediction(ts$pred, ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.5,1.7),main="ROC for Random Forest")



ccs$Class=as.integer(ccs$Class)
table(ccs$Class)
ccs$Class=ifelse(ccs$Class==2,1,0)
str(ccs)
set.seed(2341)
library(caTools)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)
library(neuralnet)
set.seed(21)
nn1 <- neuralnet(formula = Class~.,
                 data=tr,
                 hidden = 8,
                 err.fct = "sse",
                 linear.output = FALSE,
                 act.fct = "logistic",
                 lifesign = "full",
                 lifesign.step = 10,
                 threshold = 0.09,
                 stepmax = 1000)
plot(nn1)
r=compute(nn1,ts)
ts$prob=r$net.result
head(ts)
ts$pred=ifelse(ts$prob>0.5,1,0)
ts$pred=as.factor(ts$pred)
table(ts$pred)
ts$Class=as.factor(ts$Class)

library(caret)
confusionMatrix(ts$pred,ts$Class,positive = "1")

library(ROCR)
ROCRpred = ROCR::prediction(ts$prob,ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.5,1.7),main="ROC for Neural net")


##KNN
set.seed(1000)
library(caTools)
ccs$Class=as.factor(ccs$Class)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)
set.seed(200)
ctrl=trainControl(method="cv",number=10)
knn=train(Class~., data = tr, method = "knn", trControl = ctrl,tuneGrid = expand.grid(k = c(3,5,7,9,11,13,15,17,19)))
knn
plot(knn)


ts$pred=predict(knn,ts,type="prob")[,"1"]
ts$prob=ifelse(ts$pred>0.4,1,0)
ts$prob=as.factor(ts$prob)
table(ts$prob)
table(ts$Class)

library(caret)
confusionMatrix(ts$prob,ts$Class,positive = "1")

library(ROCR)
ROCRpred =ROCR::prediction(ts$pred, ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.5,1.7),main="ROC for Random Forest")


library(xgboost)
library(caTools)
set.seed(502)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)
set.seed(1243)
gd_features_train<-as.matrix(tr[,-30])
gd_label_train<-as.matrix(tr[,30])
gd_features_test<-as.matrix(ts[,-30])
xgb.fit <- xgboost(
  data = gd_features_train,
  label = gd_label_train,
  eta = 0.01,
  max_depth =15,
  min_child_weight = 10,
  nrounds = 50,
  nfold = 10,
  objective = "binary:logistic", 
  verbose = 0,               
  early_stopping_rounds = 10,
  
)

ts$pred=predict(xgb.fit, gd_features_test)

ts$prob=ifelse(ts$pred>0.4,1,0)
ts$prob=as.factor(ts$prob)


library(caret)
confusionMatrix(ts$prob,ts$Class,positive = "1")

library(ROCR)
ROCRpred =ROCR::prediction(ts$pred, ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.1),text.adj=c(-0.5,1.7),main="ROC for Xgboost")




##SVM
library(caTools)
set.seed(502)
spl=sample.split(ccs$Class,SplitRatio = 0.7)
tr=subset(ccs,spl==T)
ts=subset(ccs,spl==F)
set.seed(40001)
ctrl=trainControl(method="cv",number=10,classProbs = T)
sv=train(make.names(Class)~., data = tr, method = "svmRadial", trControl = ctrl,tunlength=10)
sv
plot(sv)
sv$bestTune
ts$pred=predict(sv,ts,type="prob")[,"X1"]
ts$prob=ifelse(ts$pred>0.5,1,0)
ts$prob=as.factor(ts$prob)
caret::confusionMatrix(ts$prob,ts$Class,positive="1")
library(ROCR)
ROCRpred = ROCR::prediction(ts$pred, ts$Class)
as.numeric(performance(ROCRpred, "auc")@y.values)
perf = performance(ROCRpred, "tpr","fpr")
plot(perf,col="black",lty=2, lwd=2)
plot(perf,lwd=3,colorize = TRUE)
plot(perf,colorize=T,print.cutoffs.at=seq(0,1,0.2),text.adj=c(-0.2,1.7),main="ROC for SVM")

q()




