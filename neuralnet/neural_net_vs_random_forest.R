library(ISLR)
library(caTools)
library(neuralnet)
library(randomForest)

df <- read.csv('bank_note_data.csv')

# Check the head and structure of the data
head(df)
str(df)
set.seed(101)
split <- sample.split(df$Class, SplitRatio = 0.70)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)
#create a neural network
nn <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,
                data = train, hidden = 10, linear.output = FALSE)

predicted.nn.values <- compute(nn, test[, 1:4])$net.result
predictions <- sapply(predicted.nn.values, round)
confusion_nn <- table(predictions, test$Class)
print("Confusion Matrix - Neural Network:")
print(confusion_nn)

# Convert Class column to factor
df$Class <- factor(df$Class)

set.seed(101)
split <- sample.split(df$Class, SplitRatio = 0.70)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# Create random forest model
model <- randomForest(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy, data = train)
rf.pred <- predict(model, test)
confusion_rf <- table(rf.pred, test$Class)
print("Confusion Matrix - Random Forest:")
print(confusion_rf)

