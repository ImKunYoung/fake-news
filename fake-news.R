library(tm)
library(SnowballC)
library(lsa)
library(caret)

# Read csv file
fake.news.df <- read.csv("data/FakeNews.csv")

# Dimensional Reduction 50% of the data
fake.news.df <- fake.news.df[1:round(0.5*nrow(fake.news.df)),]

# remove fake.news.df$real is not 1 or 0
fake.news.df<-fake.news.df[fake.news.df$label==1|fake.news.df$label==0,]

# remove na
fake.news.df<-na.omit(fake.news.df)

# remove fake.news.df$title is empty
fake.news.df<-fake.news.df[fake.news.df$title!="",]

# combine fake.news.df$title and fake.news.df$author
fake.news.df$combined <- paste(fake.news.df$title, fake.news.df$author, sep = " ")

# Calculate the number of rows
row_size <- nrow(fake.news.df)

# Create corpus from data frame
corpus <- VCorpus(VectorSource(fake.news.df[1:row_size, 6]))

# Create label
label <- as.factor(fake.news.df[1:row_size, 5])

# Tokenization
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)

# Stopwords
corpus <- tm_map(corpus, removeWords, stopwords("english"))

# Stemming
corpus <- tm_map(corpus, stemDocument)

# Compute TF-IDF
tdm <- TermDocumentMatrix(corpus)

# Initialize an empty data frame to store the results
if (any(apply(tdm, 2, sum) == 0)) {
  tdm <- tdm[, apply(tdm, 2, sum) != 0]
}

tridf <- weightTfIdf(tdm)

# Initialize an empty data frame to store the results
results.df <- data.frame(dims=integer(), tn=integer(), fn=integer(), fp=integer(), tp=integer(), accuracy=numeric(), stringsAsFactors = FALSE)

for (i in 20:21) {

  print(i)

  # Extract (20~21) concepts
  lsa.tfidf <- lsa(tridf, dims = i)

  # Convert to data frame
  words.df <- as.data.frame(as.matrix(lsa.tfidf$dk))

  # Set seed
  set.seed(123)

  # Sample 60% of the data for training
  training <- sample(row_size, 0.6 * row_size)

  # Run logistic model on training
  trainData <- cbind(label = label[training], words.df[training,])
  reg <- glm(label ~ ., data = trainData, family = binomial)

  # Compute accuracy on validation set
  validData <- cbind(label = label[-training], words.df[-training,])
  pred <- predict(reg, newdata = validData, type = "response")

  # Produce confusion matrix
  cm <- confusionMatrix(table(ifelse(pred > 0.5, 1, 0), validData$label))

  print(cm)

  # Append the results to the data frame
  results.df <- rbind(results.df, c(i, cm$table[1], cm$table[2], cm$table[3], cm$table[4], cm$overall[1]))

}

# Print the data frame
print(results.df)