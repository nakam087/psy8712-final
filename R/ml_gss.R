# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #setting path for project
library(tidyverse) #used to  clean data + visualize
library(haven) #used to read-in data
library(caret) #used for supervised machine learning algorithms 

# Data Import and Cleaning
#reading in gss2022 data
gss_import_tbl <- read_sas("../data/gss2022.sas7bdat") #used read_sas for the sas file because it supports sas7bdat files
  
#decided to only focus on a subset of the data because I ran into some data-related errors in the project that I could not fix
#included variables that relate to how income relates to political views + perceptions of others/own life
#took out variables with 75% missingness, as to not include variables with little information
gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < .75 * nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric))%>%
  select(CONINC, #total family income, adjust for inflation
         PARTYID, #republican, democrat, independent, what?
         HAPPY,#are you very happy, pretty happy, or not too happy?
         LIFE, #In general, do you find life exciting, pretty routine, or dull?
         FAIR, #Do you think most people would try to take advantage of you if they got a chance, or would they try to be fair?
         TAX, #Do you consider the amount of federal income tax which you have to pay as too high, about right, or too low?
         CLASS)%>%#would you say you belong in: the lower class, the working class, the middle class, or the upper class?
  filter(!is.na(CONINC))#taking out NAs in predictor
         

# Visualization
#creating a histogram of personal income to display distribution of family income
ggplot(gss_tbl,
       aes(x=CONINC)) + 
  geom_histogram(fill="green")+ #green like money!!
  labs(x="Total family income adjusted for inflation", #making axis labels
       y="Frequency",
       title="Total Family Income from GSS 2022")

# Analysis
#creating holdout indeces; 25:75 partition
holdout_indices <- createDataPartition(gss_tbl$CONINC,
                                       p = .25,
                                       list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,] #test is 25% of dataset
training_tbl <- gss_tbl[-holdout_indices,] #training on 75% of dataset

training_folds <- createFolds(training_tbl$CONINC) #for every combo of 9 folds, we create a model + predict value for 10th, keeping fold number consistent to compare algos

# note that for all of the machine learning methods, I used the same method for each model to compare what algorithm "worked" the best out of the 4. Thus, I have included comments explaining the decisions made in the first model, which apply to he following models. 
#machine learning models were chosen to look at a number of different approaches to machine learning linear regression 

#training lm method
#chose this because it is a more statistical approach and is an unbiased model (compared to other predictive models I have included that add bias)
model1 <- train(
  CONINC ~ ., #regressing CONRINC on everything
  training_tbl, #using training dataset
  method="lm", #standard regression method
  na.action = na.pass, #handling missing predictor data 
  preProcess = c("center","scale","zv","nzv","medianImpute"), #including centering/standardizing, deletion of non-varying or nearly 0 predictors, assuming MCAR
  trControl = trainControl(method="cv", #resampling to cross valididation (assess how generalizable 1 of the 9 folds are)
                           number=10, #number of folds
                           verboseIter=T, #more detailed info printed in the training process
                           indexOut = training_folds) #specified training folds 
)
#Warning message:
#In predict.lm(modelFit, newdata) : prediction from rank-deficient fit; attr(*, "non-estim") has doubtful cases
model1
#saving kfold cv results
#machine learning is designed to better predict "true" variance, use adjusted r^2 to account for many predictors 
cv_m1 <- model1$results$Rsquared
#calculating holdout cv results
holdout_m1 <- cor(
  predict(model1, test_tbl, na.action = na.pass), #using model to predict test data
  test_tbl$CONINC
)^2

#training elastic net model
#This was chosen because it is a hybrid of lasso and ridge regression, which helps it balance between feature selection + feature preservation
model2 <- train(
  CONINC ~ .,
  training_tbl,
  method="glmnet",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
model2
#saving kfold cv results
cv_m2 <- max(model2$results$Rsquared)
#calculating holdout cv results
holdout_m2 <- cor(
  predict(model2, test_tbl, na.action = na.pass),
  test_tbl$CONINC
)^2

#training random forests
#chose random forests because it combines multiple decision trees + refines through pruning, pretty versatile + effective even with default hyperparameter tuning

model3 <- train(
  CONINC ~ .,
  training_tbl,
  method="ranger",
  na.action = na.pass,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
model3
#saving kfold cv results
cv_m3 <- max(model3$results$Rsquared)
#calculating holdout cv results
holdout_m3 <- cor(
  predict(model3, test_tbl, na.action = na.pass),
  test_tbl$CONINC
)^2

#training extreme gradient boosting
#this was chosen because it works similarly to random forest in terms of combining decision trees, but it does this through boosting, in which trees are built sequentially to reduce the errors of the previous tree.
model4 <- train(
  CONINC ~ .,
  training_tbl,
  method="xgbLinear",
  na.action = na.pass,
  tuneLength = 1,
  preProcess = c("center","scale","zv","nzv","medianImpute"),
  trControl = trainControl(method="cv", 
                           number=10, 
                           verboseIter=T, 
                           indexOut = training_folds)
)
model4
#saving kfold cv results
cv_m4 <- max(model4$results$Rsquared)
#calculating holdout cv results
holdout_m4 <- cor(
  predict(model4, test_tbl, na.action = na.pass),
  test_tbl$CONINC
)^2

#This is a comparison of the performances of the models on the training data 
summary(resamples(list(model1, model2, model3, model4)), metric="Rsquared")
#This is the visualized comparison of each model as a dotplot
dotplot(resamples(list(model1, model2, model3, model4)), metric="Rsquared")

# Publication
function_proj <- function (x) { #creating a function where it is rounded to 2 digits and the leading 0 digit is removed in the decimal
  x <- formatC(x, format="f", digits=2)
  x <- str_remove(x, "^0")
  return(x)
}

#creating a table that describes compares the R^2 of the algos
#saving kfold cv results
#saving holdout cv results
table1_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  cv_rqs = c(
    function_proj(cv_m1),
    function_proj(cv_m2),
    function_proj(cv_m3),
    function_proj(cv_m4)
  ),
  ho_rqs = c(
    function_proj(holdout_m1),
    function_proj(holdout_m2),
    function_proj(holdout_m3),
    function_proj(holdout_m4)
  )
)
