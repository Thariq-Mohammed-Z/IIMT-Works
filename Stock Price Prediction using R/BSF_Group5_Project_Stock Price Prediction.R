# Run this line if you are running it for the first time
# install.packages(c("quantmod","forecast","ggplot2","caTools","caret","pROC","lmtest"))

library(quantmod)
library(forecast)
library(ggplot2)
library(caTools)
library(caret)
library(pROC)
library(lmtest)

# Defining function to find optimal cutoff value
find_optimal_cutoff <- function(y_true, y_pred, metric = "confusionMatrix") {
  cutoffs <- seq(0.01, 0.99, by = 0.01) # Creating a sequence of potential cutoff values
  performance <- sapply(cutoffs, function(cutoff) {
    y_pred_bin <- ifelse(y_pred > cutoff, 1, 0)
    # Ensure the levels of the factors are the same
    y_pred_bin <- factor(y_pred_bin, levels = c(0, 1))
    y_true <- factor(y_true, levels = c(0, 1))
    cm <- confusionMatrix(y_pred_bin, y_true)
    accuracy <- cm$overall["Accuracy"]
  })
  optimal_cutoff <- cutoffs[which.max(performance)]
  return(optimal_cutoff)
}

# Step 1: Data Collection and Pre-processing

selected_stock <- "IOC.NS"  # Defining the stock
getSymbols(selected_stock, src = "yahoo", from = "1994-08-11", to = Sys.Date())

# Accessing the specific column for closing prices
data <- get(selected_stock)[, paste(selected_stock, ".Close", sep = "")]
data <- na.approx(data)  # Handling missing values using linear interpolation
colnames(data) <- "Close"  # Renaming column for ease

# Adding Days column and create lag features
data$Days <- 1:nrow(data)
data$Lag1 <- lag(data$Close, 1)
data$Lag2 <- lag(data$Close, 2)

# Removing the first two rows because Lag1 and Lag2 will have NA values
data <- na.omit(data)

# Step 2: Linear Regression

# Splitting the data into training and test sets
set.seed(123)
intrain_lm <- createDataPartition(data$Close, p = 0.70, list = FALSE)
training_lm <- data[intrain_lm,]
testing_lm <- data[-intrain_lm,]

# Linear Model Development
lm_model <- lm(Close ~ Days, data = training_lm)
summary(lm_model)  # Check the R-squared for goodness of fit

# Prediction using Linear Regression
pred_lm <- predict(lm_model, newdata = testing_lm)

# Plotting Actual vs Predicted
ggplot() +
  geom_line(data = testing_lm, aes(x = Days, y = Close, color = "Actual")) +
  geom_line(aes(x = testing_lm$Days, y = pred_lm, color = "Predicted")) +
  labs(title = "Linear Regression: Actual vs Predicted", x = "Days", y = "Close Price") +
  theme_minimal()

# Residual diagnostics to check assumptions
par(mfrow = c(2, 2))
plot(lm_model)  # Checking for linearity, normality of residuals, homoscedasticity

# Step 3: Logistic Regression

# Assigning values for Price_Change (1 for Bullish, 0 for Bearish)
data$Price_Change <- ifelse(diff(data$Close, lag = 1) > 0, 1, 0)
data <- data[-1,]  # Remove first row to align Price_Change

# Splitting the data again into training and test sets
set.seed(123)
intrain_log <- createDataPartition(data$Price_Change, p = 0.70, list = FALSE)
training_log <- data[intrain_log,]
testing_log <- data[-intrain_log,]

# With only partitioned testing_log, the distribution of 0s and 1s are not even (difference is >100).
# This causes the confusion matrix to take a hit while predicting
# Hence Manual oversampling is done to get a balanced training set.

# Manual oversampling for the minority class (bullish days)
minority_class <- training_log[training_log$Price_Change == 1, ]
majority_class <- training_log[training_log$Price_Change == 0, ]
num_samples_to_generate <- nrow(majority_class) - nrow(minority_class)

# Generating synthetic samples by bootstrapping the minority class
set.seed(123)
synthetic_samples <- minority_class[sample(nrow(minority_class), num_samples_to_generate, replace = TRUE), ]
balanced_training_log <- rbind(majority_class, minority_class, synthetic_samples)

# Logistic Model Development
log_model <- glm(Price_Change ~ Days + Lag1 + Lag2, family = "binomial", data = balanced_training_log)
summary(log_model)

# Prediction using Logistic Regression
pred_log <- predict(log_model, testing_log, type = "response")

# Finding the optimal cutoff value
optimal_cutoff <- find_optimal_cutoff(testing_log$Price_Change, pred_log)
print(paste("Optimal Cutoff Value:", round(optimal_cutoff, 2)))

# Converting predictions to binary with the optimal cutoff
pred_log_bin <- ifelse(pred_log > optimal_cutoff, 1, 0)

# Convert actual values to factors with same levels
pred_log_bin <- factor(pred_log_bin, levels = c(0, 1), labels = c('Bearish', 'Bullish'))
target <- factor(testing_log$Price_Change, levels = c(0, 1), labels = c('Bearish', 'Bullish'))

# Accuracy of Logistic Regression
accuracy_log <- mean(pred_log_bin == target)
print(paste("Accuracy of Logistic Regression:", round(accuracy_log * 100, 2), "%"))

# Confusion Matrix
conf_matrix <- confusionMatrix(pred_log_bin, target)
print(conf_matrix)

# Ensuring Price_Change is a numeric vector
data_roc <- roc(as.numeric(testing_log$Price_Change), as.numeric(pred_log))

# ROC and AUC Calculation
par(mfrow = c(1,1))
plot(data_roc)
print(paste("AUC:", auc(data_roc)))

# Step 4: ARIMA Model

# ARIMA model development
arima_model <- auto.arima(data$Close, seasonal = FALSE)
summary(arima_model)

# Forecasting for the same number of periods as the testing set
forecast_horizon <- nrow(testing_lm)  # Set the forecast horizon to the number of days in the test set
forecast_arima <- forecast(arima_model, h = forecast_horizon)

# Converting the actual data from xts to numeric
actual_close <- as.numeric(testing_lm$Close)

# Plot of forecast vs actual closing prices
autoplot(forecast_arima) +
  geom_line(aes(x = 1:length(actual_close), y = actual_close, color = "Actual")) +
  labs(title = "ARIMA Forecast vs Actual", y = "Close Price", x = "Time") +
  theme_minimal()

# Accuracy of ARIMA model by comparing forecasts with test data
accuracy_arima <- accuracy(forecast_arima$mean, actual_close)
print(paste("MAE of ARIMA:", round(accuracy_arima['Test set', 'MAE'], 2)))

# Residual diagnostics for ARIMA
checkresiduals(arima_model)

# Step 5: Next Day Prediction and Bullish/Bearish Forecast

# LINEAR REGRESSION NEXT DAY PREDICTION
next_day_lm <- data.frame(Days = max(data$Days) + 1)
next_day_lm$predicted_price <- predict(lm_model, next_day_lm)
print(paste("Linear Regression Next Day Predicted Price:", round(next_day_lm$predicted_price, 2)))

# LOGISTIC REGRESSION NEXT DAY PREDICTION
# Prepare data for the next day's prediction
next_day_log <- data.frame(
  Days = max(data$Days) + 1,               # Next day
  Lag1 = tail(data$Close, 1),              # Today's close as Lag1
  Lag2 = tail(data$Close, 2)[1]            # Yesterday's close as Lag2
)

# Setting the column names explicitly
colnames(next_day_log)[2:3] <- c("Lag1", "Lag2")

# Predict the probability of the next day being bullish or bearish
next_day_log_prob <- predict(log_model, next_day_log, type = "response")

# Determine if the next day is bullish or bearish based on the optimal cutoff
next_day_log_prediction <- ifelse(next_day_log_prob > optimal_cutoff, "Bullish", "Bearish")
print(paste("Logistic Regression Prediction for the next day is:", next_day_log_prediction))

# ARIMA Next Day Prediction
next_day_arima <- forecast(arima_model, h = 1)
print(paste("ARIMA Next Day Predicted Price:", round(next_day_arima$mean, 2)))

# ARIMA Next Day Prediction (Bullish/Bearish)
if (as.numeric(next_day_arima$mean) > as.numeric(tail(data$Close, 1))) {
  print("ARIMA Prediction: Bullish")
} else {
  print("ARIMA Prediction: Bearish")
}