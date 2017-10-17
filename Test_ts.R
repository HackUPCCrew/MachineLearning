install.packages('lubridate')
install.packages("lme4")
install.packages('fpc')
library(lmer)
library(xts)

data <- read.csv('/Users/krishnakalyan3/MOOC/MachineLearning/data/test.csv')

cols <- c('Id', 'Dates', 'X', 'Y')
data_pred <- subset(data, select=cols)
head(data_pred)


data_pred$Dates <- as.POSIXlt(data_pred$Dates, "%Y-%m-%d h:m:s")
(data_pred$Dates[2])
library(lubridate)
t.lub <- ymd_hms(data_pred$Dates)

# Hours
h.str <- as.numeric(format(t.lub, "%H")) + as.numeric(format(t.lub, "%M"))/60
h.str <- as.integer(h.str)
h.str <- as.factor(h.str)

# Weekend
day.str <- as.factor(weekdays(as.Date(t.lub)))
summary(day.str)

final_data <- data.frame(h.str, day.str)
names(final_data) <- c('hours', 'weekend')
head(final_data)
data_pred_v1 <- data.frame(data_pred, final_data)
str(data_pred_v1)


data_sub <- final_data[1:100,]
str(data_sub)

formula <- cbind(lat, long) ~ hours + weekend

model <- glm(formula, final_data)
model
predict(model, final_data[1:100,])


# GeoHash
install.packages('geohash')
install.packages('doMC')

# Hashing
library(doMC)
library('geohash')
registerDoMC(4)
gh_encode(lat = -122.3996, lng = 37.73505, precision = 2)
gh_decode("hb4b4")

# Clustering
library(fpc)
db_scan <-dbscan(d, eps = 0.009, MinPts = 3)
table(db_scan$cluster)


z <- cbind(data_pred$X,  data_pred$Y)
k_means <- kmeans(z, 10)
table(k_means$cluster)
plot(z, col=k_means$cluster, pch=20)

out_data <- data.frame(final_data, k_means$cluster)
head(out_data)
out_data$hours
data_pred_v2 <- data.frame(data_pred_v1, k_means$cluster)
str(data_pred_v2)
table(data_pred_v2$k_means.cluster)

cols <- c('X', 'Y', 'k_means.cluster')
data_pred <- subset(data_pred_v2, select=cols)
write.csv(data_pred, file = "mongo_data.csv",row.names=FALSE)

str(data_pred_v2)
write.csv(data_pred_v2, file = "crime_data",row.names=FALSE)

