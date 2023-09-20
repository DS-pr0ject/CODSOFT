# Load necessary libraries
library(dplyr)
library(ggplot2)
library(caTools)
library(ROSE)

# Read the dataset
creditcard <- read.csv("C:\\Users\\Tayyaba\\Documents\\internship tasks\\task 3\\creditcard.csv")

# Data exploration and summary
dim(creditcard) # Get dimensions of the dataset
head(creditcard)# Display the first few rows of the dataset
tail(creditcard) # Display the last few rows of the dataset

# Check if there are any missing values in the dataset
any(is.na(creditcard))

# Display the structure of the dataset
str(creditcard)

#Summary
table(creditcard$Class)            # Count the number of each class (fraudulent vs. non-fraudulent)
names(creditcard)                  # List column names
summary(creditcard)                # Display summary statistics
sd(creditcard$Time)                # Calculate standard deviation of 'Time'
var(creditcard)                    # Calculate variance of all numeric columns


# Scale the data to remove extreme values
# Data preprocessing
head(creditcard,6)
creditcard$Amount=scale(creditcard$Amount)
NewData=creditcard[,-1]
head(NewData)


# Data visualization
hist(NewData$Class)                 # Plot histogram of the 'Class' variable
hist(NewData$Amount[NewData$Amount < 100])  # Plot histogram of 'Amount' with a filter

# Create scatter plot with color differentiation by class
ggplot(creditcard, aes(x = Amount, y = Time, color = factor(Class))) +
  geom_point() +
  labs(title = "Credit Card Fraud Detection",
       x = "Amount",
       y = "Time",
       color = "Class") +
  theme_minimal()

# Create bar plot to show counts of fraud and non-fraud transactions
ggplot(NewData, aes(x = factor(Class))) +
  geom_bar(position = "dodge", fill = "orange") +
  labs(title = "Credit Card Fraud Counts",
       x = "Class",
       y = "Percentage") +
  scale_x_discrete(labels = c("Not Fraud", "Fraud")) +
  theme_minimal()

# More data visualization - histogram of 'Time' for each class
ggplot(creditcard, aes(x = Time , fill = factor(Class))) + 
  geom_histogram(bins = 100) + 
  labs(x = "Time elapsed since first transcation (seconds)", 
       y = "no. of transactions", 
       title = "Distribution of transactions across time") +
  facet_grid(Class ~ ., scales = 'free_y') + theme()


# Model the data to train data and test data
set.seed(123)
data_sample = sample.split(NewData$Class,SplitRatio=0.80)
train_data = subset(NewData,data_sample==TRUE)
test_data = subset(NewData,data_sample==FALSE)
dim(train_data)
dim(test_data)


#Fit the model using logistic regression, with family as binomial
Logistic_Model = glm(Class~., test_data,family = binomial())
summary(Logistic_Model)
plot(Logistic_Model)
# Make predictions on the test data
lr.predict <- predict(Logistic_Model,train_data, probability = TRUE)
auc.gbm = roc(test_data$Class, lr.predict, plot = TRUE, col = "yellow")

over_sampled_data <- ovun.sample(Class ~ ., data = train_data, method = "over", N = nrow(train_data[grep("Class", colnames(train_data)), ])$Class)

# Load required libraries for modeling
library(glmnet)
library(ROCR)

Logistic_Model = glm(Class~., train_data,family = binomial())
summary(Logistic_Model)


# Convert probabilities to class labels (0 or 1)
predicted_class <- ifelse(lr.predict > 0.5, 1, 0)

# Calculate the confusion matrix
confusion_matrix <- table(predicted_class, test_data$Class)

# Calculate precision, recall, and F1-score
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print the results of evaluation metrics
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1-score:", f1_score, "\n")




