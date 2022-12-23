library(tm)
library(SnowballC)
library(lsa)
library(caret)
library(dplyr)
library(plyr)
library(ggplot2)

# Read csv file
fake.news.df<-read.csv("data/FakeNews.csv")

# Remove rows with missing values
fake.news.df<-fake.news.df[complete.cases(fake.news.df),]

# Remove non-specific data
fake.news.df<-fake.news.df[fake.news.df$real %in% c(1, 0),]

windows()

ggplot(fake.news.df, aes(x=real)) + geom_bar()




# library(naivebayes)
# NBmodel <- naive_bayes(X_train, y_train)
# predicted <- predict(NBmodel, X_test)
# predicted
#
# library(caret)
# cm2 <- confusionMatrix(y_test, predicted)
# score2 <- accuracy(y_test, predicted)
# cr2 <- classification_report(y_test, predicted)
#
# library(ROCR)
# pred <- prediction(predicted, y_test)
# perf <- performance(pred, "auc")
# auc2 <- as.numeric(perf@y.values)

# Calculate the number of rows
row_size<-nrow(fake.news.df)

# Create corpus from data frame
corpus<-VCorpus(VectorSource(fake.news.df[1:row_size, 1]))

# Create label
label<-as.factor(fake.news.df[1:row_size, 5])

# Tokenization
corpus<-tm_map(corpus, stripWhitespace)
corpus<-tm_map(corpus, removePunctuation)
corpus<-tm_map(corpus, removeNumbers)

# Stopwords
corpus<-tm_map(corpus, removeWords, stopwords("english"))

# Stemming
corpus<-tm_map(corpus, stemDocument)

# Compute TF-IDF
tdm<-TermDocumentMatrix(corpus)

if (any(apply(tdm, 2, sum)==0)) {
  tdm<-tdm[, apply(tdm, 2, sum)!=0]
}

tridf<-weightTfIdf(tdm)

# Extract (20) concepts
lsa.tfidf<-lsa(tridf, dims=20)

# Convert to data frame
words.df<-as.data.frame(as.matrix(lsa.tfidf$dk))

# Set seed
set.seed(123)

# Sample 60% of the data for training
training<-sample(row_size, 0.6*row_size)

# Run logistic model on training
trainData<-cbind(label=label[training], words.df[training,])
reg<-glm(label ~ ., data=trainData, family=binomial)

# Compute accuracy on validation set
validData<-cbind(label=label[-training], words.df[-training,])
pred<-predict(reg, newdata=validData, type="response")

# Produce confusion matrix


cm <- print(confusionMatrix(table(ifelse(pred>0.5, 1, 0), validData$label)))
print(cm)
# 교차 검증을 위한 패키지 설치
install.packages("caret")
library(caret)

# # 나이브 베이즈 분류기를 적용합니다.
# nb <- naiveBayes(label ~ ., data=trainData)
#
# # 검증 데이터 셋에 대해 예측을 수행합니다.
# pred <- predict(nb, newdata=validData)
#
# # 예측 결과와 실제 값을 비교하여 정확도를 계산합니다.
# accuracy <- mean(pred == validData$label)
# print(paste("Accuracy:", accuracy))
#
# # 예측 결과와 실제 값을 비교한 결과를 출력합니다.
# confusionMatrix(table(pred, validData$label))


# 결정 트리를 적용할 수 있는 패키지 설치
install.packages("rpart")
library(rpart)

ipred_installed <- require("ipred", character.only=TRUE)
print(ipred_installed)

rpart_installed <- require("rpart", character.only=TRUE)
print(rpart_installed)

# rpart 패키지가 설치되어 있는지 확인합니다.

install.packages("rpart")
library(rpart)

install.packages("ipred")
library(ipred)
# 결정 트리를 적용합니다.
# 이 코드는 이전에 제공한 코드에서 사용한 trainData와 validData를 사용합니다.
dt <- rpart(label ~ ., data=trainData)

# 검증 데이터 셋에 대해 예측을 수행합니다.
pred <- predict(fake.news.df, newdata=validData, type="class")

# 예측 결과와 실제 값을 비교하여 정확도를 계산합니다.
accuracy <- mean(pred == validData$label)
print(paste("Accuracy:", accuracy))

# 예측 결과와 실제 값을 비교한 결과를 출력합니다.
cm <- confusionMatrix(table(pred, validData$label))

print(cm)