#Project Objective
#It is unrealistic to expect to be able to predict all flight delays with a high probability. There
#are simply too many unpredictable factors (such as weather). However, you should define
#the problem as a classification problem and use the classification techniques that we've
#discussed to obtain useful information. By this I mean that the end result should be
#something that a traveler might use to reduce the chances of ending up on the delayed
#flight. This may be a probability estimate of experiencing a delayed flight, identifying which
#factors contribute to flight delay, and/or identifying scenarios where travel should be
#avoided. I don't know exactly what you can find in this data, but the objective is to find
#patterns that can be translated into useful information for travelers.

library(caret)
library(RWeka)
library(e1071)
library(Boruta)
library(klaR)
library(DMwR) #for smote
library(randomForest)
library(pROC)

flights <- read.csv("C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/flights.csv")
airlines <- read.csv("C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/airlines.csv")
airports <- read.csv("C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/airports.csv")



#gives a range of values and returns 1,163,816 values that can be input into the next line
#Alternative would be to add replace = TRUE then certain values may appear more than once
#this would lead to the flights2 sample data set having duplicate instances pulled from flights data set
set.seed(1234)
x <- sample(1:2500000,581908) 

memory.limit
memory.limit(size = 4007) #changed memory size from 4007 default to 8000
flights2 <- flights[x,]

ls() # indicates which objects are taking up space
rm(flights) #selects object to remove from memory

summary(flights)

#data$AIRLINE_DELAY=is.no(data$AIRLINE_DELAY)] <-0 #replaces data points with some specified value
#data$Class<-ifelse(data$AIRLINE_DELAY>0,c("yes"),c("No")) #if else to create a new class label column

summary(airlines)
summary(airports)

filter.flights <- flights2[flights2$DIVERTED == 1,]
summary(filter.flights)


str(flights2)
########################################preparing data for Boruta and feature selection####################

library(tidyr)
flights2$Air_FN <- paste(flights2$AIRLINE, flights2$FLIGHT_NUMBER, sep='')# COMBINES AIRLINE AND FLIGHT NUMBER

flights2$AM_PM_Departure<-ifelse(flights2$DEPARTURE_TIME>1200,c('PM'),c('AM')) #INDICATES AM/PM DEPARTURE
flights2$AM_PM_Arrival<-ifelse(flights2$ARRIVAL_TIME>1200,c('PM_Arrival'),c('AM_Arrival')) #INDICATES AM/PM ARRIVAL


flights2 <- flights2[-1] #YEAR
flights3 <- cbind(flights2[1],flights2[3],flights2[7:8],flights2[11:19],flights2[22],flights2[26:33]) #CREATES UPDATED TABLE

library(qdapTools)
airports2 <- cbind(airports[1],airports[8])
flights3$Origin_Airport_region <- lookup(flights3$ORIGIN_AIRPORT, airports2) #add region column based on origin airport
flights3$Destination_Airport_region <- lookup(flights3$DESTINATION_AIRPORT, airports2) #add region column based on origin airport
flights3$Direction <- paste(flights3$Origin_Airport_region, flights3$Destination_Airport_region, sep='_to_')# add column indicating direction of flight
flights3$class_delayed <- ifelse(flights3$ARRIVAL_DELAY>15,c('Yes'),c('No')) #class label indicating whether there is a delay or not


#check for unique values in origin ariport column of flights 3
#Origin_Airports <- data.frame(unique(flights3$ORIGIN_AIRPORT))
#write.csv(Origin_Airports,file = "C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/OriginAirports.csv")
str(flights3)

#flights4 contains derived attributes and removed attributes with class label of class_delayed
flights4 <- cbind(flights3[1:2],flights3[5:13],flights3[20:26])

flights4 <- na.omit(flights4) #removes NA's from data.frame
flights4$class_delayed <- factor(flights4$class_delayed) #changes class label to factor
flights4$Direction <- factor(flights4$Direction)
summary(flights4)

write.csv(flights4,"C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/10%_Train1.csv")

#set.seed(1234)

#Data Partition into test and train sets and benchmark model with no attribute selection or subsampling
#trainIndex <- createDataPartition(flights4$class_delayed, p=.67,list = FALSE,times = 1)
#flights4train.bm <- flights4[trainIndex,]
#flights4test.bm <- flights4[-trainIndex,]

ctrl<-trainControl(savePred=T,classProb=T, sampling = "up")
treeGrid <- expand.grid(C=(2:10)*0.05, M = (1:10)*1) # added the tuning parameter of M (not in the notes)
mod3 <- train(class_delayed~.,data=flights4,method="J48",tuneGrid=treeGrid)
summary(mod3)
#mod3
mod3.predict <- predict(mod3,flights4train.bm)
confusionMatrix(mod3.predict,flights4train.bm$class_delayed, mode = "prec_recall")
plot(mod3)

mod3$bestTune
mod3$finalModel
mod3.predict <- predict(mod3,flights4test.bm)
confusionMatrix(mod3.predict,flights4test.bm$class_delayed, mode = "prec_recall")
mod3$modelInfo
mod3$finalModel
predictors(mod3)

#############################################################################################
#############################################################################################
#############################################################################################
#Boruta Attribute Selection using Random Forest
Boruta.flights4 <- Boruta(class_delayed ~ ., data = flights4, doTrace = 2, ntree = 500, maxRuns = 100) #generates importante attributes
Boruta.Credit
plot(Boruta.Credit) #plots important attributes
Boruta.Credit.TenativeFix <- TentativeRoughFix(Boruta.Credit)#accounts for tentative attributes by median z scores
Boruta.Credit.TenativeFix
#plotZHistory(Boruta.CreditTrain.TenativeFix)
Boruta.Credit.Confirmd <- getConfirmedFormula(Boruta.Credit.TenativeFix) #generates model based on confirmed attributes
Boruta.Credit.Confirmd
attStats(Boruta.Credit.TenativeFix) #displays data frame of attribute selections --mean, min, max, decision

#Creation of data.frame with Boruta identified attributes
Boruta.Credit <- cbind(Credit[,1:8],Credit[10],Credit[,13:15],Credit[21])
str(Boruta.Credit)

#Data Partition into test and train sets
trainIndex <- createDataPartition(Boruta.Credit$class, p=.67,list = FALSE,times = 1)
CreditTrain <- Boruta.Credit[trainIndex,]
CreditTest <- Boruta.Credit[-trainIndex,]

###########################################Model Training on Reduced Data Set#####################################
####################################################################################################################
####################################################################################################################


flights <- read.csv("C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/ReducedDataSet.csv")

summary(flights)
file.info("C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/ReducedDataSet.csv")
colnames(flights)
str(flights)

flights <- cbind(flights[2:4],flights[8:10],flights[14:19])
str(flights)

flights$DEPARTURE_DELAY<-ifelse(flights$DEPARTURE_DELAY>0,c('Depart_Delayed'),c('Depart_Early')) #INDICATES DEPARTURE Delay
flights$ELAPSED_TIME<-ifelse(flights$ELAPSED_TIME>118,c('>117'),c('<118')) #Elapsed time
flights$AIR_TIME<-ifelse(flights$AIR_TIME>93,c("Long Flight"),c("Short Flight")) #long or short flight
flights$DISTANCE<-ifelse(flights$DISTANCE>648,c(">648 miles"),c("<648 miles")) #miles of flight

flights$DEPARTURE_DELAY <- factor(flights$DEPARTURE_DELAY)
flights$ELAPSED_TIME <- factor(flights$ELAPSED_TIME)
flights$AIR_TIME <- factor(flights$AIR_TIME)
flights$DISTANCE <- factor(flights$DISTANCE)



set.seed(1234)
#x <- sample(1:285685,285685)
#x <- sample(1:142842,142842)
x <- sample(1:285685,25000)

memory.limit
memory.limit(size = 100000) #changed memory size from 4007 default to 8000
flights.Train <- flights[x,]

x <- sample(285685:571369,25000)
flights.Test <- flights[x,]

write.csv(flights.Train,"C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/Train2.csv")
write.csv(flights.Test,"C:/Users/JHeuer/Documents/Jacey/Business Analytics Courses/IE 583/Final Project/Test2.csv")

memory.limit
memory.limit(size = 100000)

#Training caret Random Forest method with ROC metric
ctrl<-trainControl(sampling = "down",savePred=T,classProb=T, search = "random", summaryFunction=twoClassSummary)
metric <- "ROC"
mtry <- sqrt(ncol(flights.Train))
#tunegrid <- expand.grid(.mtry=(2:10))
tunegrid <- expand.grid(.mtry=mtry)
flights.rf <- train(class_delayed~., data=flights.Train, method= "rf", metric=metric, tuneGrid=tunegrid, trControl = ctrl, tuneLength = 15)
print(flights.rf)
#summary(rf_default)
plot(flights.rf)
flights.rf.predict <- predict(flights.rf,flights.Train)
confusionMatrix(flights.rf.predict,flights.Train$class_delayed, mode = "prec_recall")

flights.rf.predict <- predict(flights.rf,flights.Test)
confusionMatrix(flights.rf.predict,flights.Train$class_delayed, mode = "prec_recall")
#modup2$modelInfo
#modup2$finalModel
#predictors(modup2)
flights.rf$contrasts

#create AUC for caret Random Forest
rf2.1_default.predict <- predict(rf2.1_default,Credit2, type = "prob")
rf2.1_default.predict
rpartROC <- roc(Credit2$class, rf2.1_default.predict[, "bad"])
plot(rpartROC, type = "S", print.thres = .5)
rpartROC

summary(flights.Train)
#decision tree
ctrl<-trainControl(sampling = "down",savePred=T,classProb=T, summaryFunction=twoClassSummary)
metric <- "ROC"
treeGrid <- expand.grid(C=(2:10)*0.05, M = (1:5)*1) # added the tuning parameter of M (not in the notes)
moddown <- train(class_delayed~., data=flights.Train,method="J48",trControl=ctrl,tuneGrid=treeGrid, metric = metric)
summary(moddown)
moddown$finalModel
moddown$bestTune
#mod3
moddown.predict <- predict(moddown,flights.Train)
confusionMatrix(moddown.predict,flights.Train$class_delayed, mode = "prec_recall")
#plot(mod3)

moddown$bestTune
moddown$finalModel
moddown.predict <- predict(moddown,flights.Test)
confusionMatrix(moddown.predict,flights.Train$class_delayed, mode = "prec_recall")
moddown$modelInfo
moddown$finalModel
predictors(moddown)

#create AUC for caret Random Forest
moddown.predict <- predict(moddown,flights.Test, type = "prob")
moddown.predict
rpartROC <- roc(flights.Train$class_delayed, moddown.predict[, "Yes"])
plot(rpartROC, type = "S", print.thres = .5)
rpartROC

