# Spam Filter using Leazy Learning (Non-parametric Classification)
# Use k-nearest neighbors algorithm in class package 
#
# Data
# Response type   :  0,1 = not-spam, spam classes (categorial)
# Class proportion:  not-spam(60%), spam(40%)
# Predictor types :  numerical 
#
# Model Fitting
# Method 1: using class package
# Method 2: using RWeka package




# Part 1: Install appropriate packages  
#
#install.packages('class')
#install.packages('RWeka')



# Part 2: Read data files
#
# Read predictors and response variable
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
#cor(data)
#pairs(data[,41:50], cex=0.5, upper.panel=NULL)


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
library(class)
cat('                                            ', '\n')
cat('--------------------------------------------', '\n')
cat('Using class package ------------------------', '\n')
cat('                                            ', '\n')


# 4a: Cross validation to get the best k
Acc = vector()  # overall accuracy 
Knn = vector()  # number of nearest neighbors 

# Set the seeds
set.seed(100)
# get the vertical size of the data frame 
n <- nrow(df_n)

for (nn in 1:15)
{
# Separate instances into training set(90%) and test set (10%)
train_size <- sample(1:n, floor(0.90*n))         
# split data
train_data  <- df_n[ train_size, ]
train_class <- train_data[, c(58)]

# Default leave-one-out cross validation 
cvKNN <- knn.cv(train_data[ ,-c(58)], cl=train_class, k=nn, prob=FALSE)
# Confusion table 
tab   <-  table(pred=cvKNN, obsn=train_class)
tab   <- round(prop.table(tab) * 100, digits=3)

Acc[nn] <- (tab[1]+tab[4])
Knn[nn] <- nn
} 
# Visualize cross-validation output 
plot(Knn, Acc, ylim=c(70,95), cex=0.85, pch=16,
	       ylab='Accuracy',
	       xlab='Number of Nearest Neighbors',
	       main='Cross-validation Ouput (k-NN)')
# ----------------------------------------------------------------


# 4b: Best k value found. Fit a model without rebalancing the class proportion 
# Set the seeds
set.seed(200)
# Get the vertical size of the data frame
n <- nrow(df_n)
# Separate instances into training set(90%) and test set (10%)
train_size <- sample(1:n, floor(0.90*n))   
train_data <- df_n[ train_size, ]
test_data  <- df_n[-train_size, ]

train_class <- train_data[, c(58)]
test_class  <-  test_data[, c(58)]

predKNN <- knn(train_data[, -c(58)], test_data[ ,-c(58)], cl=train_class, k=1)

# Print results
cat(' ', '\n')
tab  <-  table(pred=predKNN, obsn=test_class)
tab  <- round(prop.table(tab) * 100, digits=3)
cat('Overall Accuracy without Rebalancing: ', tab[1]+tab[4], '%', '\n')
cat(' ','\n')
# ----------------------------------------------------------------


# 4c: Best k value found. Fit a model with rebalancing the class proportion
#set.seed(300)
Acc = vector()  # overall accuracy of a given sample
Fnn = vector()  # false negative of the given sample
Fpp = vector()  # false positive of the given sample
Nrb = vector()  # number of times class proportion has been rebalanced


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
nrs  <- sample( 1:nr1, floor(frn*nr1) ) # number of rows sampled 
d3   <- d1[nrs, ]                       # balanced data

# class is rebalanced 
df_r <- rbind(d2,d3)
# get the vertical size of the data frame  
n <- nrow(df_r)
# Separate instances into training set(90%) and test set (10%)
train_size <- sample(1:n, floor(0.90*n))   
train_data <- df_r[ train_size, ]
test_data  <- df_r[-train_size, ]

train_class <- train_data[ ,c(58)]
test_class  <-  test_data[ ,c(58)]

predKNN <- knn(train_data[, -c(58)], test_data[ ,-c(58)], cl=train_class, k=1)

# print model performance
# confusion table 
tab  <-  table(pred=predKNN, obsn=test_class)
# accuracy
AC   <- round( ((tab[1]+tab[4])/sum(tab)) * 100, digits=3 )
# false negative (not-spam email predicted as spam)
FN   <- round(   (tab[2]/(tab[1]+tab[2])) * 100, digits=3 )
# false positive (spam predicted as not-spam email)
FP   <- round(   (tab[3]/(tab[3]+tab[4])) * 100, digits=3 )
#cat('Trial: ' ,i, 'Accuracy: ' ,AC, '& False Negative Rate: ' ,FN, '%', '\n')

Acc[i] <- AC
Fnn[i] <- FN
Fpp[i] <- FP
Nrb[i] <- i
}

#cat('', '\n')
cat('Overall Accuracy, FNR, FPR: ', mean(Acc), mean(Fnn), mean(Fpp), '%', '\n')
cat('', '\n')


# Visualize model performance as a function of class rebalance
png("fig_modelPerformance_KNN.png", width=800, height=600)
par(mfrow=c(2,1))
plot(Nrb, Acc, ylim=c(70,100), cex=0.85, pch=16, 
               cex.axis=0.85, cex.lab=0.85, cex.main=0.95,
	       xlab='Class Rebalance Trails',
	       ylab='Accuracy (percent)',
	       main='Model Performance and Class Rebalance (k-NN)'
	       )

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






## Method 2:
## Using "RWeka" package 
library(RWeka)
cat('                                            ', '\n')
cat('--------------------------------------------', '\n')
cat('Using RWeka package ------------------------', '\n')
# Find the best value for K, between 1 and 11 
# Use 10-fold cross validation to evaluate our classifier
seed <- 1234
fit_classifier <- IBk(email_type ~ ., data = df_n, 
                      control = Weka_control(K = 11, X = TRUE))

eval_classifier <- evaluate_Weka_classifier(fit_classifier,
					    newdata = NULL, 
					    numFolds = 10, 
					    cost=matrix(c(0,1,1,0), ncol=2), 
 					    seed=seed, 
					    complexity = TRUE,
                              		    class = TRUE)
# Print results
cat('Printing Overall Results --------------------', '\n')
print(eval_classifier)

cat('', '\n')
cat('Printing Overall Accuracy -------------------, '\n'')
tab <- (eval_classifier$confusionMatrix)
tab <- t(tab)
tab <- round(prop.table(tab) * 100, digits=3)
cat('Overall Accuracy: ', tab[1]+tab[4], '%', '\n')


