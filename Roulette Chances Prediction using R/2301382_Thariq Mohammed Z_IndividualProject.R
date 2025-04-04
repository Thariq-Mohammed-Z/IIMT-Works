# Loading Required Libraries
# install.packages(c("dplyr", "MASS", "nnet", "car", "AER", "margins", "ggplot2"))
library(dplyr)
library(MASS)
library(nnet)
library(AER)
library(margins)
library(ggplot2)

# Loading the Dataset
data <- read.csv("../../Project/Individual/roulette_100000_rounds.csv")
str(data)
head(data)

# ---------------------------------------------------------------------------------------
# Binary Choice Models (Logit and Probit)
# ---------------------------------------------------------------------------------------
# Question: What factors influence the probability of a bet on red winning?

# Preparing data to create independent variables
data$Winning_Range <- cut(data$Winning.Number, breaks = c(-1, 12, 24, 36), labels = c("Low", "Medium", "High"))
data$Even_Odd <- ifelse(data$Winning.Number %% 2 == 0, "Even", "Odd")

# Logit Model
logit_model <- glm(Red.Bet.Win ~ Winning_Range + Even_Odd, family = binomial(link = "logit"), data = data)
summary(logit_model)

# Probit Model
probit_model <- glm(Red.Bet.Win ~ Winning_Range + Even_Odd, family = binomial(link = "probit"), data = data)
summary(probit_model)

# Marginal Effects for Logit
logit_marginal <- margins(logit_model)
summary(logit_marginal)

# Marginal Effects for Probit
probit_marginal <- margins(probit_model)
summary(probit_marginal)

# ---------------------------------------------------------------------------------------
# Count Data Models
# ---------------------------------------------------------------------------------------
# Question: How does the number of wins vary across colors in a given period?

# Aggregating the data for count analysis
data_summary <- data %>%
  group_by(Winning.Color) %>%
  summarise(Count = n())

# Poisson Model
poisson_model <- glm(Count ~ Winning.Color, family = poisson, data = data_summary)
summary(poisson_model)

# ---------------------------------------------------------------------------------------
# Ordered Regression Models (Ordered Probit and Ordered Logit)
# ---------------------------------------------------------------------------------------
# Question: Does the range of winning numbers (low, medium, high) depend on previous round results?

# Preparing the data for Lag in winning range as independent variable
data <- data %>%
  mutate(Prev_Winning_Range = lag(Winning_Range))

# Ordered Probit Model
ordered_probit <- polr(Winning_Range ~ Prev_Winning_Range, data = data, method = "probit")
summary(ordered_probit)

# Ordered Logit Model
ordered_logit <- polr(Winning_Range ~ Prev_Winning_Range, data = data, method = "logistic")
summary(ordered_logit)

# Brant Test for Ordered Logit
# install.packages("brant")
library(brant)
brant(ordered_logit)

# ---------------------------------------------------------------------------------------
# Multinomial Regression Models (Logit and Probit)
# ---------------------------------------------------------------------------------------
# Question: What factors influence the probability of a specific color (red, black, or zero) being the winner?

# Multinomial Logit Model
multinomial_logit <- multinom(Winning.Color ~ Prev_Winning_Range, data = data)
summary(multinomial_logit)

# Multinomial Probit Model
multinomial_probit <- multinom(Winning.Color ~ Prev_Winning_Range, data = data, method = "probit")
summary(multinomial_probit)

# ---------------------------------------------------------------------------------------
# Truncated/Censored Regression Models
# ---------------------------------------------------------------------------------------
# Question: Can we predict the winning number given that it falls within a specific range (e.g., > 18)?

# Using Truncated Data by filering rows with Winning Number > 18
truncated_data <- subset(data, Winning.Number > 18)

# Least Squares on Full Sample
lm_full <- lm(Winning.Number ~ Red.Bet.Win + Black.Bet.Win + Even.Bet.Win + Odd.Bet.Win, data = data)
summary(lm_full)

# Least Squares on Truncated Sample
lm_truncated <- lm(Winning.Number ~ Red.Bet.Win + Black.Bet.Win + Even.Bet.Win + Odd.Bet.Win, data = truncated_data)
summary(lm_truncated)

# Tobit Model
tobit_model <- tobit(Winning.Number ~ Red.Bet.Win + Black.Bet.Win + Even.Bet.Win + Odd.Bet.Win, left = 18, data = data)
summary(tobit_model)

# ---------------------------------------------------------------------------------------
# Diagnostics for Checking the Assumptions
# ---------------------------------------------------------------------------------------
# Heteroskedasticity: Breusch-Pagan Test
# install.packages("lmtest")
library(lmtest)
bptest(logit_model)

# Multicollinearity: Variance Inflation Factor (VIF)
vif(logit_model)

# ---------------------------------------------------------------------------------------
# Model Performance Evaluation
# ---------------------------------------------------------------------------------------
# Splitting the data into training (80%) and testing (20%)
set.seed(123)  # For reproducibility
train_indices <- sample(1:nrow(data), size = 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Predictive performance for Logit Model
logit_model_train <- glm(Red.Bet.Win ~ Winning_Range + Even_Odd, family = binomial(link = "logit"), data = train_data)
logit_predictions <- predict(logit_model_train, newdata = test_data, type = "response")
logit_class <- ifelse(logit_predictions > 0.5, 1, 0)
confusion_matrix_logit <- table(Predicted = logit_class, Actual = test_data$Red.Bet.Win)
print("Confusion Matrix for Logit Model:")
print(confusion_matrix_logit)

# Predictive performance for Probit Model
probit_model_train <- glm(Red.Bet.Win ~ Winning_Range + Even_Odd, family = binomial(link = "probit"), data = train_data)
probit_predictions <- predict(probit_model_train, newdata = test_data, type = "response")
probit_class <- ifelse(probit_predictions > 0.5, 1, 0)
confusion_matrix_probit <- table(Predicted = probit_class, Actual = test_data$Red.Bet.Win)
print("Confusion Matrix for Probit Model:")
print(confusion_matrix_probit)

# Predictive performance for Multinomial Logit Model
multinomial_logit_train <- multinom(Winning.Color ~ Prev_Winning_Range, data = train_data)
multinomial_predictions <- predict(multinomial_logit_train, newdata = test_data)
confusion_matrix_multinomial <- table(Predicted = multinomial_predictions, Actual = test_data$Winning.Color)
print("Confusion Matrix for Multinomial Logit Model:")
print(confusion_matrix_multinomial)

# Predictive performance for Truncated Model (using RMSE)
predicted_lm <- predict(lm_full, newdata = test_data)
rmse_lm <- sqrt(mean((test_data$Winning.Number - predicted_lm)^2))
print("RMSE for Truncated Model:")
print(rmse_lm)

# ---------------------------------------------------------------------------------------
# Visualizations for Business Questions
# ---------------------------------------------------------------------------------------

# Binary Choice Models: Logit predictions for "Red Bet Win"
ggplot(data = test_data, aes(x = Winning_Range, y = logit_predictions)) +
  geom_boxplot(aes(fill = Winning_Range)) +
  labs(title = "Probability of Red Bet Win by Winning Range", y = "Predicted Probability", x = "Winning Range") +
  theme_minimal()

# Count Data Models: Wins by Color
color_counts <- data %>% group_by(Winning.Color) %>% summarise(Count = n())
ggplot(color_counts, aes(x = Winning.Color, y = Count, fill = Winning.Color)) +
  geom_bar(stat = "identity") +
  labs(title = "Count of Wins by Color", y = "Count", x = "Winning Color") +
  theme_minimal()

# Ordered Regression Models: Predictions by Previous Winning Range
ggplot(data = test_data, aes(x = Prev_Winning_Range, fill = Winning_Range)) +
  geom_bar(position = "dodge") +
  labs(title = "Predicted Range of Winning Numbers by Previous Range", x = "Previous Winning Range", y = "Count") +
  theme_minimal()

# Multinomial Regression Models: Actual vs Predicted Winning Colors
ggplot(test_data, aes(x = Winning.Color, fill = factor(multinomial_predictions))) +
  geom_bar(position = "dodge") +
  labs(title = "Actual vs Predicted Winning Colors", x = "Actual Color", fill = "Predicted Color") +
  theme_minimal()

# Truncated Regression Models: Predicted vs Actual Winning Numbers
rmse_plot <- ggplot(test_data, aes(x = Winning.Number, y = predicted_lm)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Winning Numbers", x = "Actual", y = "Predicted") +
  theme_minimal()
print(rmse_plot)