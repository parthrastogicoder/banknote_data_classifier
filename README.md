# Bank Note Classifier: README

## Overview

This project utilizes machine learning techniques to classify banknotes as authentic or forged based on features extracted from images of the banknotes. The dataset, `bank_note_data.csv`, includes attributes such as variance, skewness, kurtosis, and entropy of the images. The primary objective is to build predictive models using Neural Networks and Random Forest classifiers to accurately identify the authenticity of banknotes.

## Requirements

- R
- R Libraries: 
  - `caTools`
  - `neuralnet`
  - `randomForest`

## Setup Instructions

1. **Install Required Libraries**
   Open your R console or RStudio and install the necessary libraries:
   ```R
   install.packages("caTools")
   install.packages("neuralnet")
   install.packages("randomForest")
   ```

2. **Load the Data**
   Ensure the `bank_note_data.csv` file is in your working directory. The dataset contains the following columns:
   - `Image.Var`: Variance of the image
   - `Image.Skew`: Skewness of the image
   - `Image.Curt`: Kurtosis of the image
   - `Entropy`: Entropy of the image
   - `Class`: Class label (0 for authentic, 1 for forged)

## Code Explanation

1. **Load Libraries**
   ```R
   library(caTools)
   library(neuralnet)
   library(randomForest)
   ```

2. **Read the Data**
   ```R
   df <- read.csv('bank_note_data.csv')
   ```

3. **Check the Data Structure**
   ```R
   head(df)
   str(df)
   ```

4. **Data Splitting**
   Split the data into training and testing sets with a 70-30 ratio:
   ```R
   set.seed(101)
   split <- sample.split(df$Class, SplitRatio = 0.70)
   train <- subset(df, split == TRUE)
   test <- subset(df, split == FALSE)
   ```

5. **Neural Network Model**
   - **Create Neural Network**
     ```R
     nn <- neuralnet(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy,
                     data = train, hidden = 10, linear.output = FALSE)
     ```
   - **Make Predictions**
     ```R
     predicted.nn.values <- compute(nn, test[, 1:4])$net.result
     predictions <- sapply(predicted.nn.values, round)
     confusion_nn <- table(predictions, test$Class)
     print("Confusion Matrix - Neural Network:")
     print(confusion_nn)
     ```

6. **Random Forest Model**
   - **Convert Class Column to Factor**
     ```R
     df$Class <- factor(df$Class)
     ```
   - **Re-split the Data**
     ```R
     set.seed(101)
     split <- sample.split(df$Class, SplitRatio = 0.70)
     train <- subset(df, split == TRUE)
     test <- subset(df, split == FALSE)
     ```
   - **Create Random Forest Model**
     ```R
     model <- randomForest(Class ~ Image.Var + Image.Skew + Image.Curt + Entropy, data = train)
     rf.pred <- predict(model, test)
     confusion_rf <- table(rf.pred, test$Class)
     print("Confusion Matrix - Random Forest:")
     print(confusion_rf)
     ```

## Results

- **Confusion Matrix for Neural Network**
  The confusion matrix shows the performance of the Neural Network in classifying banknotes:
  ```R
  print("Confusion Matrix - Neural Network:")
  print(confusion_nn)
  ```

- **Confusion Matrix for Random Forest**
  The confusion matrix shows the performance of the Random Forest classifier:
  ```R
  print("Confusion Matrix - Random Forest:")
  print(confusion_rf)
  ```

## Conclusion

This project demonstrates the use of Neural Networks and Random Forest classifiers to predict the authenticity of banknotes. The confusion matrices provide insights into the classification accuracy and performance of the models.
