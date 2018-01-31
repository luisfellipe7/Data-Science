# load in packages we'll use
library(tidyverse) # utility functions
library(rpart) # for regression trees
library(randomForest) # for random forests

# read the data and store data in DataFrame titled melbourne_data
melbourne_data <- read_csv(".../melb_data.csv")

#printing a summary of the data
summary(melbourne_data)

#building a decision tree based on the dataset
fit <- rpart(Price ~ Rooms + Bathroom + Landsize + BuildingArea +
               YearBuilt + Lattitude + Longtitude, data = melbourne_data)

#looking at the tree and making it bigger
plot(fit, uniform=TRUE)
text(fit, cex=.6)

#Now the fitted model can be used to predict house prices
print("Making predictions for the following 5 houses:")
print(head(melbourne_data))

print("The predictions are")
print(predict(fit, head(melbourne_data)))

print("Actual price")
print(head(melbourne_data$Price))
