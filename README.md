---
title: "SaiGiridharSakethBangaru_CINF624_Final"
output: html_document
date: "2024-05-03"
---


# 1. Introduction

The objective of this analysis is to develop a predictive model for breast cancer diagnosis using machine learning techniques. The dataset utilized is the Breast Cancer Wisconsin (Diagnostic) Dataset obtained from Kaggle, which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses. The main techniques applied in this analysis are Ridge and Lasso regression, which are forms of linear regression with regularization to prevent over-fitting and improve model generalization.

```{r}
library(dplyr)
library(readr)

getwd()
setwd("C:/Users/sai98/OneDrive/Desktop/624/Final Assignment")
```

```{r}
options(repos = "https://cloud.r-project.org")
install.packages("png")
```

```{r}
bcp_df <- 
  read.csv(
    file = "Breast Cancer Prediction.csv")
```

```{r}
bcp_df <- select(bcp_df,-id,-X)
```

```{r}
head(bcp_df)
summary(bcp_df)
```

```{r}
install.packages("DataExplorer")

library(DataExplorer)
```

# 2.  Methodology

## Data Preprocessing

In the data preprocessing step, the following actions were taken:

-   Handling missing values: Any missing values in the dataset were identified and either imputed or removed, depending on the extent of missingness and the impact on the analysis.

-   Encoding categorical variables: If the dataset contained categorical variables, they were encoded using techniques such as one-hot encoding to convert them into numerical format suitable for modeling.

-   Normalization or scaling features: Continuous features were normalized or scaled to ensure that all variables have a similar scale, which can improve the convergence and performance of the regression models.

```{r}
library(dplyr)
bcp_df <- bcp_df %>% relocate(diagnosis,.after= fractal_dimension_worst)
```
```{r}
#check for missing variables
sapply(bcp_df, function(x) sum(is.na(x)))
```
In the results displayed, you can see the data has 569 records, each with 31 columns.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

```{r}
## Create a frequency table
diagnosis.table <- table(bcp_df$diagnosis)
colors <- terrain.colors(2) 
# Create a pie chart 
diagnosis.prop.table <- prop.table(diagnosis.table)*100
diagnosis.prop.df <- as.data.frame(diagnosis.prop.table)
pielabels <- sprintf("%s - %3.1f%s", diagnosis.prop.df[,1], diagnosis.prop.table, "%")

pie(diagnosis.prop.table,
  labels=pielabels,  
  clockwise=TRUE,
  col=colors,
  border="gainsboro",
  radius=0.8,
  cex=0.8, 
  main="frequency of cancer diagnosis")
legend(1, .4, legend=diagnosis.prop.df[,1], cex = 0.7, fill = colors)
```
```{r}
bcp_df$diagnosis <- factor(bcp_df$diagnosis, levels = c("M","B"), labels = c(1,0))
```

```{r}
bcp_df$diagnosis <- as.character(bcp_df$diagnosis)

bcp_df$diagnosis <- as.numeric(bcp_df$diagnosis)
```


```{r}
# Install and load necessary packages
if (!requireNamespace("corrplot", quietly = TRUE)) {
  install.packages("corrplot")
}
library(corrplot)
corMatMy <- cor(bcp_df[,1:30])
corrplot(corMatMy, order = "hclust", tl.cex = 0.7)
```

# 3. Model Implementation

## Split data into training and test sets

The implementation of Ridge and Lasso regression models involved the following steps:

-   Splitting the dataset into training and testing sets: The dataset was divided into a training set, used to train the models, and a testing set, used to evaluate their performance.

-   Implementing Ridge regression: Ridge regression was implemented using the glmnet package in R. K-fold cross-validation was applied to find the optimal regularization parameter (alpha) that minimizes the mean squared error.

-   Implementing Lasso regression: Similarly, Lasso regression was implemented using the glmnet package with k-fold cross-validation to find the optimal regularization parameter.

-   Training the final models: Once the optimal hyperparameters were determined, the final Ridge and Lasso models were trained on the entire training set.

We will Split the available data into a training set and a testing set. (70% training, 30% test)

```{r}
library(tidyverse)
library(ISLR)
install.packages("caret")
library(caret)
```

```{r}
# Split the dataset into training and testing sets
set.seed(123) # For reproducibility
trainIndex <- createDataPartition(bcp_df$diagnosis, p = 0.7, 
                                  list = FALSE)
train_data <- bcp_df[trainIndex, ]
test_data <- bcp_df[-trainIndex, ]
```

```{r}
# Implement Ridge Regression
install.packages("glmnet")
library(glmnet)

# Check for missing values in the training dataset
missing_values <- any(is.na(train_data))
if (missing_values) {
  stop("Missing values found in the training dataset. Please handle missing values before proceeding.")
}

# Check if the response variable (diagnosis) is constant
constant_response <- length(unique(train_data$diagnosis)) == 1
if (constant_response) {
  stop("Response variable (diagnosis) is constant. Ridge regression fails at standardization step.")
}

ridge_model <- cv.glmnet(as.matrix(train_data[, -c(1, 2)]), 
                         train_data$diagnosis, 
                         alpha = 0, # Ridge regression
                         nfolds = 10)
plot(ridge_model)
```

```{r}
# Find the optimal lambda/alpha for Ridge Regression
optimal_alpha_ridge <- ridge_model$lambda.min

```

```{r}
# Train the final Ridge model
library(glmnet)
final_ridge_model <- glmnet(as.matrix(train_data[, -c(1, 2)]), 
                            train_data$diagnosis, 
                            alpha = 0, # Ridge regression
                            lambda = optimal_alpha_ridge)
```

```{r}
# Implement Lasso Regression
lasso_model <- cv.glmnet(as.matrix(train_data[, -c(1, 2)]), 
                         train_data$diagnosis, 
                         alpha = 1, # Lasso regression
                         nfolds = 10)
```

```{r}
plot(lasso_model)
```
```{r}
# Find the optimal lambda/alpha for Lasso Regression
optimal_alpha_lasso <- lasso_model$lambda.min
```

```{r}
# Train the final Lasso model
final_lasso_model <- glmnet(as.matrix(train_data[, -c(1, 2)]), 
                            train_data$diagnosis, 
                            alpha = 1, # Lasso regression
                            lambda = optimal_alpha_lasso)
```

# 4. Applying machine learning models

In this section we will:

1.Train the algorithm on the first part,

2.make predictions on the second part and

3.evaluate the predictions against the expected results.

## Define K-fold cross-validation control objects
```{r}
kfold_ctrl <- trainControl(method = "repeatedKFold", repeats = 5)
tenfold_ctrl <- trainControl(method = "repeatedKFold", repeats = 10)
```
## Random Forest
```{r}
selected_features <- c("radius_worst", "area_worst", "perimeter_worst", "concave.points_worst", "concave.points_mean", "concavity_worst", "radius_mean","perimeter_mean","concavity_mean","area_se")
```


```{r}
X_train <- train_data[, selected_features]
y_train <- train_data$diagnosis
X_test <- as.matrix(select(test_data,radius_worst, area_worst, perimeter_worst, concave.points_worst, concave.points_mean, concavity_worst, radius_mean,perimeter_mean,concavity_mean,area_se))

```


```{r}
install.packages("randomForest")
library(randomForest)

rf_model <- randomForest(x = X_train, 
                         y = as.factor(y_train), 
                         ntree = 500,                
                         mtry = sqrt(length(selected_features)),  
                         importance = TRUE)
```


```{r}
predictions <- predict(rf_model, newdata = X_test)
confusion_matrix_rf <- table(predictions, test_data$diagnosis)
confusion_matrix_rf

accuracy_rf <- sum(diag(confusion_matrix_rf)) / sum(confusion_matrix_rf)
print(paste("Accuracy:",accuracy_rf*100))
```
```{r}
var_importance <- importance(rf_model)
print(var_importance)
```
## SVM

```{r}
install.packages("e1071")
library(e1071)

svm_model <- svm(x = X_train, 
                         y = as.factor(y_train), 
                         ntree = 500,                
                         mtry = sqrt(length(selected_features)),  
                         importance = TRUE)

predictions1 <- predict(svm_model, newdata = X_test)
confusion_matrix_svm <- table(predictions1, test_data$diagnosis)
confusion_matrix_svm

accuracy_svm <- sum(diag(confusion_matrix_svm)) / sum(confusion_matrix_svm)
print(paste("Accuracy:",accuracy_svm*100))
```
## XGBoost model

```{r}
install.packages("xgboost")
library(xgboost)
```


```{r}
X_train <- as.matrix(train_data[, selected_features])
y_train <- train_data$diagnosis

xgb_model <- xgboost(data = X_train, 
                      label = y_train, 
                      nrounds = 100,  # Number of boosting rounds (adjust as needed)
                      objective = "binary:logistic",  # Binary classification
                      eval_metric = "error",  # Evaluation metric (error rate)
                      max_depth = 3,  # Maximum tree depth (adjust as needed)
                      eta = 0.3)  # Learning rate (adjust as needed)

predictions <- predict(xgb_model, newdata = X_test, type = "response")

# Convert probabilities to binary predictions (0 or 1)
predictions <- ifelse(predictions > 0.5, 1, 0)

# Evaluate model performance
confusion_matrix_xgb <- table(predictions, test_data$diagnosis)
print(confusion_matrix_xgb)

accuracy_xgb <- sum(diag(confusion_matrix_xgb)) / sum(confusion_matrix_xgb)
print(paste("Accuracy:", accuracy_xgb*100))
```

## Naive bayes

```{r}
install.packages("naivebayes")
library(naivebayes)
```


```{r}
nb_model <- naive_bayes(x = X_train, 
                         y = as.factor(y_train), 
                         ntree = 500,                
                         mtry = sqrt(length(selected_features)),  
                         importance = TRUE)

predictions2 <- predict(nb_model, newdata = X_test)
confusion_matrix_nb <- table(predictions2, test_data$diagnosis)
confusion_matrix_nb

accuracy_nb <- sum(diag(confusion_matrix_nb)) / sum(confusion_matrix_nb)
print(paste("Accuracy:",accuracy_nb*100))
```
# 5. Model Evaluation

The performance of the Ridge and Lasso regression models was evaluated using the following metrics:

-   RMSE (Root Mean Squared Error): This metric measures the average magnitude of the errors between predicted and actual values.

-   R² score (Coefficient of Determination): This metric quantifies the proportion of the variance in the target variable that is predictable from the independent variables.

These metrics were calculated using the predictions of the trained models on the testing set, providing insights into their predictive accuracy and goodness of fit.

```{r}
# Evaluate the models on the test set
ridge_predictions <- predict(final_ridge_model, s = optimal_alpha_ridge, 
                             newx = as.matrix(test_data[, -c(1, 2)]))
lasso_predictions <- predict(final_lasso_model, s = optimal_alpha_lasso, 
                             newx = as.matrix(test_data[, -c(1, 2)]))
```

```{r}
# Calculate evaluation metrics for Ridge regression
ridge_rmse <- sqrt(mean((ridge_predictions - test_data$diagnosis)^2))
ridge_r_squared <- cor(ridge_predictions, test_data$diagnosis)^2

# Calculate evaluation metrics for Lasso regression
lasso_rmse <- sqrt(mean((lasso_predictions - test_data$diagnosis)^2))
lasso_r_squared <- cor(lasso_predictions, test_data$diagnosis)^2
```

```{r}
# Print the evaluation metrics
cat("Ridge Regression Metrics:\n")
cat("RMSE:", ridge_rmse, "\n")
cat("R-squared:", ridge_r_squared, "\n\n")

cat("Lasso Regression Metrics:\n")
cat("RMSE:", lasso_rmse, "\n")
cat("R-squared:", lasso_r_squared, "\n\n")
```

```{r}
# Define the model names and their corresponding accuracies
models <- c("Random Forest", "SVM", "XGBoost", "Naive Bayes")
accuracies <- c(accuracy_rf*100, accuracy_svm*100, accuracy_xgb*100, accuracy_nb*100)

# Create a data frame for plotting
accuracy_df <- data.frame(Model = models, Accuracy = accuracies)

# Plotting the bar graph
library(ggplot2)

# Set colors for the bars
colors <- c("navyblue","forestgreen", "orange", "darkgrey")

# Create the bar plot with accuracy annotations
ggplot(accuracy_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = paste(round(Accuracy, 1), "%")), 
            position = position_stack(vjust = 0.5), 
            size = 4, color = "black", fontface = "bold") +
  labs(x = "Model", y = "Accuracy (%)", fill = "Model") +
  ggtitle("Comparison of Model Accuracies") +
  scale_fill_manual(values = colors) +  # Apply custom colors
  theme_minimal()  # Set a minimalistic theme
```

# 6. Results

## Exploratory Data Analysis

Upon exploring the Breast Cancer Wisconsin dataset, several insights were gleaned:

-   The dataset contains a mix of numerical and categorical variables, with the target variable being the diagnosis (M = malignant, B = benign).

-   Features such as mean radius, mean texture, and mean perimeter exhibit varying distributions, with some features showing clear differences between malignant and benign cases.

-   Correlation analysis suggests strong correlations between certain features, indicating potential multicollinearity issues that could impact model performance.

## Ridge and Lasso Regression Models

The Ridge and Lasso regression models were fitted to the dataset, with k-fold cross-validation used to determine optimal hyper parameters.

The following were the main findings: - Ridge Regression: The optimal regularization parameter (alpha) was found to be 0.047655380793354, resulting in a trained model with coefficients penalized to prevent overfitting.

-   Lasso Regression: The optimal alpha value for Lasso regression was 0.0138917963741767, indicating a model with sparse coefficients due to feature selection.

## Model Evaluation

The performance metrics (RMSE and R² score) for both Ridge and Lasso regression models on the test set are as follows:

-   Ridge Regression:
    -   RMSE: 0.06890879
    -   R² score: 0.9839018
-   Lasso Regression:
    -   RMSE: 0.01496427
    -   R² score: 1

These metrics provide insights into the predictive accuracy and goodness of fit of the models, with lower RMSE values and higher R² scores indicating better performance which suggests that the Lasso Regression has more predictive accuracy and has better fit.

The best results for accuracy (detection of breast cases) is Random Forest with 94.7%.

# 7. CONCLUSION
This analysis explored the application of machine learning techniques for breast cancer diagnosis using the Breast Cancer Wisconsin dataset. The analysis focused on Ridge and Lasso regression models with regularization to address overfitting and improve model generalization. Additionally, Random Forest, SVM, XGBoost and Naive Bayes models were implemented and evaluated.

Key findings include:

The exploratory data analysis revealed valuable insights into the dataset characteristics, including feature distributions, potential multicollinearity, and the distribution of the target variable (diagnosis).

Ridge and Lasso regression models were implemented using k-fold cross-validation for hyperparameter tuning. Both models achieved good performance on the test set, with Lasso regression exhibiting a slight edge in terms of RMSE and R² score.

Among all implemented models, Random Forest achieved the highest accuracy of 94.7% in detecting breast cancer cases.
