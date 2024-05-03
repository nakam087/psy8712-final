# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #setting path for project
library(tidyverse) #used to  clean data + visualize
library(haven) #used to read-in data
library(caret) #used for supervised machine learning algorithms 
library(parallel)
library(doParallel)

# Data Import and Cleaning
#reading in gss2018 data -- 2022 data was not working because of new added variables 
gss_import_tbl <- read_sav("../data/GSS2018.sav")%>% #used read_sav because it specializes in sav files. the sas file has some issues that appear in the ML process
  filter(!is.na(INCOME)) %>% #inflation adjusted personal income; taking out missing values 
  select(-RINCOME,-INCOM16,-INCOME16,-RINCOM16)#removing other income related variables that might be overly related to personal income
  
#retaining variables with less than 75% missingness for prediction
#because there are a large number of variables, I have only kept variables with less than 75% missingness to include the variables with the most relevant information and ensure a better fit with my model
gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl))<.75*nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric))#making all variables numeric to use ML 

# Visualization
#creating a barplot of family income to display distribution of family income
ggplot(gss_tbl,
       aes(x=INCOME)) + 
  geom_bar(fill="green")+ #green like money!!
  labs(x="Total family income", #making axis labels
       y="Frequency",
       title="Total Family Income from GSS 2018")

# Analysis
#creating holdout indeces; 25:75 partition
holdout_indices <- createDataPartition(gss_tbl$INCOME,
                                       p = .25,
                                       list = T)$Resample1
test_tbl <- gss_tbl[holdout_indices,] #test is 25% of dataset
training_tbl <- gss_tbl[-holdout_indices,] #training on 75% of dataset

training_folds <- createFolds(training_tbl$INCOME) #for every combo of 9 folds, we create a model + predict value for 10th, keeping fold number consistent to compare algos

# note that for all of the machine learning methods, I used the same method for each model to compare what algorithm "worked" the best out of the 4. Thus, I have included comments explaining the decisions made in the first model, which apply to he following models. 
#machine learning models were chosen to look at a number of different approaches to machine learning linear regression 

#training lm method
#chose this because it is a more statistical approach and is an unbiased model (compared to other predictive models I have included that add bias)
model1 <- train(
  INCOME ~ ., #regressing CONRINC on everything
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
  test_tbl$INCOME
)^2

#training elastic net model
#This was chosen because it is a hybrid of lasso and ridge regression, which helps it balance between feature selection + feature preservation
model2 <- train(
  INCOME ~ .,
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
  test_tbl$INCOME
)^2

#training random forests
#chose random forests because it combines multiple decision trees + refines through pruning, pretty versatile + effective even with default hyperparameter tuning

#kept having internal errors, so trying clustering
local_cluster <- makeCluster(detectCores()-1) #making cluster, on my computer should be 7 cores
registerDoParallel(local_cluster) #telling following MLs to be run in parallel if possible

model3 <- train(
  INCOME ~ .,
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
#This produced the following error, but there are probably missing values in the data, as I only accounted for 75% missingness. This was because I wanted to explore how well income could predict other variables.
#In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,:There were missing values in resampled performance measures.
#saving kfold cv results
cv_m3 <- max(model3$results$Rsquared)
#calculating holdout cv results
holdout_m3 <- cor(
  predict(model3, test_tbl, na.action = na.pass),
  test_tbl$INCOME
)^2

#training extreme gradient boosting
#this was chosen because it works similarly to random forest in terms of combining decision trees, but it does this through boosting, in which trees are built sequentially to reduce the errors of the previous tree.
model4 <- train(
  INCOME ~ .,
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
  test_tbl$INCOME
)^2
stopCluster(local_cluster) #stopping parallelization
registerDoSEQ()

#This is a comparison of the performances of the models on the training data 
summary(resamples(list(model1, model2, model3, model4)), metric="Rsquared")
#This is the visualized comparison of each model as a dotplot
dotplot(resamples(list(model1, model2, model3, model4)), metric="Rsquared")

# Publication
function_proj <- function (x) { #creating a function where it is rounded to 2 digits and the leading 0 digit is removed in the decimal. This is to compare the values cleanly to one another in a table
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
table1_tbl