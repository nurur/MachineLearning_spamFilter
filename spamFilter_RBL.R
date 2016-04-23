# Spam Filter using Rule Based Learning (Non-parametric Classification)
# Use RIPPER algorithm in RWeka package 
#
# Data
# Response type   :  0,1 = not-spam, spam classes (categorial)
# Class proportion:  not-spam(60%), spam(40%)
# Predictor types :  numerical 




# Part 1: Install appropriate packages  
#
#install.packages('e1071')
#install.packages('RWeka')




# Part 2: Read data files
#
# Read predictors and response variables 
data <- read.table('spambase.data.txt', stringsAsFactors =FALSE)
x    <- seq(1, nrow(data), 1) 
FUN  <- function(x){ as.numeric(unlist( strsplit(data$V1[x], ",", "") )) }
df   <- data.frame( t(sapply(x, FUN)) )

# Read column names file as a character data frame 
names <- readLines('spambase.names.txt')




# Part 3: Explore and Clean the data 
#
#print(names)
# Get the column names of the predictors
# The column names start from row 34 and end in row 90
# Strip the data frame to cull the column (predictor) names 
x        <- seq(34,90,1)
FUN_name <- function(x){ unlist(strsplit(names[x], ":", ""))[1] }
colNames <- sapply(x, FUN_name)

# Add column name of the response var column
colNames[58] <- "email_type"

# Change the column names of df
colnames(df) <- colNames

# Change the response variable from numeric type to categorical type
df$email_type = ifelse(df$email_type==1, 'ys', 'ns')
df$email_type = factor(df$email_type)


# Check for pair-wise correlations
#cor(df)
#pairs(df[,41:50], cex=0.5, upper.panel=NULL)


# Normalize the predictors 
FUN_norm <- function(x){
            norm <- (x-min(x))/(max(x)-min(x))
	    return(norm)
            }

dNorm <- as.data.frame( lapply(df[1:57], FUN_norm) )

# Add response variable into normalized data frame 
df_n <- cbind(dNorm, email_type=df$email_type)

# get the vertical size of the data frame 
n    <- nrow(df_n)
# Shuffle the instances of the data frame
df_n <- df_n[ order(runif(n)), ] 




# Part 4: Fit a Model
library(RWeka)
cat('                                            ', '\n')
cat('--------------------------------------------', '\n')
cat('Using RWeka package ------------------------', '\n')
cat('                                            ', '\n')


# Rebalancing class proportion
set.seed(110)

Acc <- vector()   # overall accuracy of given sample
Fnn <- vector()
Fpp <- vector()
Nrb <- vector()   # number of times class proportion has been rebalanced


cat('Class Rebalance Trail', '\n')
for (i in 1:25) 
{
# balance class proportion 
d1   <- df_n[which(df_n$email_type =='ns'), ]
d2   <- df_n[which(df_n$email_type =='ys'), ]

nr1  <- nrow(d1) 
nr2  <- nrow(d2) 
# fraction of one class w.r.t. the other in the original data   
frn  <- ifelse(nr2<nr1, nr2/nr1, nr1/nr2)        
nrs  <- sample( 1:nr1, floor(frn*nr1) )  # number of rows sampled 
d3   <- d1[nrs, ]                        # data balanced in class prop

# class proportion is rebalanced 
df_r <- rbind(d2,d3)
# get the vertical size of the new data frame  
n    <- nrow(df_r)
# shuffle the instances of the data frame 
df_r <- df_r[ order(runif(n)), ] 
# Separate instances into training set(90%) and test set (10%)
train_size <- sample(1:n, floor(0.90*n))   
train_data <- df_r[ train_size, ]
test_data  <- df_r[-train_size, ]

train_class<- train_data[ ,c(58)]
test_class <-  test_data[ ,c(58)]

formula <- email_type ~ .
fitRIP  <- JRip(formula, data=train_data)
predRIP <- predict(fitRIP, newdata=test_data, type='class')



# print model performance
# confusion table
tab  <-  table(pred=predRIP, obsn=test_class)
# accuracy
AC   <- round( ((tab[1]+tab[4])/sum(tab)) * 100, digits=3 )
# false negative (not spam email predicted as spam)
FN   <- round(   (tab[2]/(tab[1]+tab[2])) * 100, digits=3 )
# false negative (not spam email predicted as spam)
FP   <- round(   (tab[3]/(tab[3]+tab[4])) * 100, digits=3 )
#cat('Trial: ', i, 'Accuracy: ', AC, '& False Negative Rate: ', FN, '%', '\n')

Acc[i] <- AC
Fnn[i] <- FN
Fpp[i] <- FP
Nrb[i] <- i
}

#cat('', '\n')
cat('Overall Accuracy, FNR, FPR: ', mean(Acc), mean(Fnn), mean(Fpp), '%', '\n')
cat('', '\n')


# Visualize model performance as a function of class rebalance
png("fig_modelPerformance_RBL.png", width=800, height=600)
par(mfrow=c(2,1))
plot(Nrb, Acc, ylim=c(70,100), cex=0.85, pch=16,
	       cex.axis=0.85, cex.lab=0.85, cex.main=0.95,
               xlab='Class Rebalance Trails', 
	       ylab='Accuracy (percent)',
	       main='Model Performance and Class Rebalance (Rule Based)')

plot(Nrb, Fnn, ylim=c(0.0,20), cex=0.85, pch=16,
	       cex.axis=0.85, cex.lab=0.75, 
               xlab='Class Rebalance Trails', 
	       ylab='False Negative Rate (percent)')

par(new=TRUE)
x = 1:length(Nrb)
y = rep(0, length(Nrb))
plot(x, y, ylim=c(0.0,20), type='l', lty=3, ylab='', xlab='', axes=FALSE)
text(5.0,0.75, 'Ideal Case', cex=0.60)
text(5.0,19, 'postive class  = not-spam', cex=0.85)
text(5.0,17, 'negative class = spam    ', cex=0.85)
text(6.5,15, 'FNR = actual not-spam predicted as spam', cex=0.85)
dev.off()




# Part 5: Evaluate Model Performance 
library("ROCR")
predR2  <- predict(fitRIP, newdata=test_data, type='probability')
predROC <- prediction(predR2[,2], test_class)
par(mfrow=c(2,1))
#plot(performance(predROC, "tpr", "fpr"))
#abline(0, 1, lty = 2)
#plot(performance(predROC, "tpr"), ylim=c(0.7, 1))
#plot(performance(predROC, "fpr"))

plot(performance(predROC, "acc"), ylim=c(0.0,0.95),
			          cex.axis=0.75, cex.lab=0.85,
				  main='Model Performance (Rule Based)'
				 )
plot(performance(predROC, "fnr"), ylim=c(0.0,0.95),
			          cex.axis=0.75, cex.lab=0.85)
