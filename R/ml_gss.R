# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #setting path for project
library(tidyverse) #used to  clean data + visualize
library(haven) #used to read-in data
library(caret) #used for supervised machine learning algorithms 

# Data Import and Cleaning
#reading in gss2022 data
gss_import_tbl <- read_sas("../data/gss2022.sas7bdat")%>% #used read_sas for the sas file because it supports sas7bdat files
  filter(!is.na(CONRINC)) %>% #inflation adjusted personal income; taking out missing values 
  select(-REALINC,-CONINC, -REALRINC, -INCOM16)#removing other income related variables that might be overly related to personal income

#retaining variables with less than 50% missing-ness for prediction
#because there are a large number of variables, I have only kept variables with less than 50% missing-ness to include the variables with the most relevant information and ensure a better fit with my model, since I am not using more advanced imputation methods.  
#some variables are also perfectly predicted by the model, which led to early errors in my code when I included less than 75% missingness.
gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl)) < .75 * nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric)) #making all variables numeric to use ML 

# Visualization
#creating a histogram of personal income to display distribution of personal income (it seems most fall around $40,000)
ggplot(gss_tbl,
       aes(x=CONRINC)) + 
  geom_histogram(fill="green")+ #green like money!!
  labs(x="Inflation adjusted personal income", #making axis labels
       y="Frequency",
       title="Histogram of Personal Income from GSS 2022")

# Analysis
#creating holdout indeces; 25:75 partition
holdout_indices <- createDataPartition(gss_tbl$CONRINC,
                                       p = .25,
                                       list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,] #test is 25% of dataset
training_tbl <- gss_tbl[-holdout_indices,] #training on 75% of dataset

training_folds <- createFolds(training_tbl$CONRINC) #for every combo of 9 folds, we create a model + predict value for 10th, keeping fold number consistent to compare algos

# note that for all of the machine learning methods, I used the same method for each model to compare what algorithm "worked" the best out of the 4. Thus, I have included comments explaining the decisions made in the first model, which apply to he following models. 
#machine learning models were chosen to look at a number of different approaches to machine learning linear regression 

#training lm method
#chose this because it is a more statistical approach and is an unbiased model (compared to other predictive models I have included that add bias)
model1 <- train(
  CONRINC ~ ., #regressing CONRINC on everything
  training_tbl, #using training dataset
  method="lm", #standard regression method
  na.action = na.pass, #handling missing predictor data 
  preProcess = c("center","scale","zv","nzv","medianImpute"), #including centering/standardizing, deletion of non-varying or nearly 0 predictors, assuming MCAR
  trControl = trainControl(method="cv", #resampling to cross valididation (assess how generalizable 1 of the 9 folds are)
                           number=10, #number of folds
                           verboseIter=T, #more detailed info printed in the training process
                           indexOut = training_folds) #specified training folds 
)
model1
#saving kfold cv results
#machine learning is designed to better predict "true" variance, use adjusted r^2 to account for many predictors 
cv_m1 <- model1$results$Rsquared
#calculating holdout cv results
holdout_m1 <- cor(
  predict(model1, test_tbl, na.action = na.pass), #using model to predict test data
  test_tbl$CONRINC
)^2

#training elastic net model
#This was chosen because it is a hybrid of lasso and ridge regression, which helps it balance between feature selection + feature preservation
model2 <- train(
  CONRINC ~ .,
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
  test_tbl$CONRINC
)^2

#training random forests
#chose random forests because it combines multiple decision trees + refines through pruning, pretty versatile + effective even with default hyperparameter tuning

#kept getting internal failures like this one, so I had to add some hyperparameters to the model to get it to run
#model fit failed for Fold01: mtry=353, min.node.size=5, splitrule=variance Error in ranger::ranger(dependent.variable.name = ".outcome", data = x,  : User interrupt or internal error

model3 <- train(
  CONRINC ~ .,
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
  test_tbl$CONRINC
)^2

#training extreme gradient boosting
#this was chosen because it works similarly to random forest in terms of combining decision trees, but it does this through boosting, in which trees are built sequentially to reduce the errors of the previous tree.
model4 <- train(
  CONRINC ~ .,
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
  test_tbl$CONRINC
)^2

#This is a comparison of the performances of the models on the training data 
summary(resamples(list(model1, model2, model3, model4)), metric="Rsquared")
#This is the visualized comparison of each model as a dotplot
dotplot(resamples(list(model1, model2, model3, model4)), metric="Rsquared")

# Publication
make_it_pretty <- function (formatme) {
  formatme <- formatC(formatme, format="f", digits=2)
  formatme <- str_remove(formatme, "^0")
  return(formatme)
}

table1_tbl <- tibble(
  algo = c("regression","elastic net","random forests","xgboost"),
  cv_rqs = c(
    make_it_pretty(cv_m1),
    make_it_pretty(cv_m2),
    make_it_pretty(cv_m3),
    make_it_pretty(cv_m4)
  ),
  ho_rqs = c(
    make_it_pretty(holdout_m1),
    make_it_pretty(holdout_m2),
    make_it_pretty(holdout_m3),
    make_it_pretty(holdout_m4)
  )
)
