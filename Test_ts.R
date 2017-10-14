install.packages('xts')
library(xts)

data <- read.csv('/Users/krishnakalyan3/MOOC/MachineLearning/data/test.csv')
cols <- c('Id', 'X', 'Y')
data$Dates
data_pred <- subset(data, select=cols)
data_pred$Dates <- as.POSIXlt(data_pred$Dates, "%Y-%m-%d h:m:s")
strptime(data_pred$Dates, "h")

str(data_pred)

install.packages('kernlab')
library('kernlab')
