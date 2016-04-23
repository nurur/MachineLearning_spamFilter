# Spam Filter using Decision Tree Based Learning (Non-parametric Classification)
# Use C5.0 algorithm in C50 package 
#
# Data
# Response type: 
# 1,0 = spam, non-spam classes (categorial)
# Predictor types:
# numerical 


library("ROCR")
library(e1071)
library(C50)
library(RWeka)



# Read predictors and response variables 
data <- read.table('spambase.data.txt', stringsAsFactors =FALSE)
x    <- seq(1, nrow(data), 1) 
FUN  <- function(x){ as.numeric(unlist( strsplit(data$V1[x], ",", "") )) }
df   <- data.frame( t(sapply(x, FUN)) )

# Read column names file as a character data frame 
names <- readLines('spambase.names.txt')


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
#set.seed(99)
df_n <- df_n[ order(runif(n)), ] 




# Rebalancing class proportion
set.seed(110)

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



## Fit a Model 
# NBC Model
formula <- email_type ~ .
fitNBC  <- naiveBayes(formula, data=train_data, laplace=1)
predNBC <- predict(fitNBC, newdata=test_data, type='raw')
predROC <- prediction(predNBC[,2], test_class)

a   <- performance(predROC, "tpr", "fpr")
fpr <- unlist( slot(a, 'x.values') )
tpr <- unlist( slot(a, 'y.values') )
thd <- unlist( slot(a, 'alpha.values') )
b   <- performance(predROC, "tnr", "fnr")
fnr <- unlist( slot(b, 'x.values') )
tnr <- unlist( slot(b, 'y.values') )

thd <- ifelse(thd>1, 1, thd)
NBC <- data.frame(tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, thd=thd)



# DTL Model (Use Boosting by seeting "trails" > 1)
fitC50  <- C5.0(x=train_data[, -58], y=train_class,
	   		       	    rules=FALSE,
				    weights=NULL,
	   		            trials=10,
				    costs=NULL,
				    control = C5.0Control(subset=TRUE,
				    bands = 0, winnow = FALSE,
				    noGlobalPruning = FALSE,
				    CF = 0.25, minCases = 2,
				    fuzzyThreshold = FALSE,
				    sample = 0,
				    seed = sample.int(4096,size=1) - 1L,
				    earlyStopping = TRUE, label = "outcome")
				    )
				    
predC50 <- predict(fitC50, newdata=test_data, type='prob')
predROC <- prediction(predC50[,2], test_class)

a   <- performance(predROC, "tpr", "fpr")
fpr <- unlist( slot(a, 'x.values') )
tpr <- unlist( slot(a, 'y.values') )
thd <- unlist( slot(a, 'alpha.values') )
b   <- performance(predROC, "tnr", "fnr")
fnr <- unlist( slot(b, 'x.values') )
tnr <- unlist( slot(b, 'y.values') )

thd <- ifelse(thd>1, 1, thd)
DTL <- data.frame(tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, thd=thd)




# RBL Model
formula <- email_type ~ .
fitRIP  <- JRip(formula, data=train_data)
predRIP <- predict(fitRIP, newdata=test_data, type='probability')
predROC <- prediction(predRIP[,2], test_class)

a   <- performance(predROC, "tpr", "fpr")
fpr <- unlist( slot(a, 'x.values') )
tpr <- unlist( slot(a, 'y.values') )
thd <- unlist( slot(a, 'alpha.values') )
b   <- performance(predROC, "tnr", "fnr")
fnr <- unlist( slot(b, 'x.values') )
tnr <- unlist( slot(b, 'y.values') )

thd <- ifelse(thd>1, 1, thd)
RBL <- data.frame(tpr=tpr, fpr=fpr, tnr=tnr, fnr=fnr, thd=thd)



# Plot
pdf("fig_rocCurve.pdf")

x=seq(0,1,0.01)
y=x
plot(x,y, type='l', lwd=0.5, lty=3, axes=FALSE, xlab='', ylab='')
par(new=TRUE)
plot(NBC$fpr, NBC$tpr, type='l', col='green', lwd=2, axes=FALSE, xlab='',ylab='')
par(new=TRUE)
plot(DTL$fpr, DTL$tpr, type='l', col='red', lwd=2, axes=FALSE, xlab='',ylab='')
par(new=TRUE)
plot(RBL$fpr, RBL$tpr, type='l', col='blue', lwd=2, xlab='FPR', ylab='TPR',
	      main='ROC Curve')


text(0.8, 0.25, 'Decision Tree (without Cost Matrix)', col='red', cex=0.75)
text(0.8, 0.22, 'Rule Based',  col='blue', cex=0.75)
text(0.8, 0.19, 'Naive Bayes', col='green', cex=0.75)

dev.off()