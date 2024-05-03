# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path)) #setting path for project
library(tidyverse) #used to  clean data + visualize, more intuitive data-cleaning
library(haven) #used to read-in data
library(caret) #used for supervised machine learning algorithms 
library(parallel) #had to parallelize data to speed up machine learning process
library(doParallel)
library(apaTables)

# Data Import and Cleaning
#reading in gss2018 data -- 2022 data was not working because of new added variables 
gss_import_tbl <- read_sav("../data/GSS2018.sav")%>% #used read_sav because it specializes in sav files. the sas file has some issues that appear in the ML process
  filter(!is.na(INCOME)) %>% #family income; taking out missing values 
  select(-RINCOME,-INCOM16,-INCOME16,-RINCOM16)#removing other income related variables that might be overly related to family income
  
#retaining variables with less than 75% missingness for prediction
#because there are a large number of variables, I have only kept variables with less than 75% missingness to include the variables with the most relevant information and ensure a better fit with my model
gss_tbl <- gss_import_tbl[, colSums(is.na(gss_import_tbl))<.75*nrow(gss_import_tbl)] %>%
  mutate(across(everything(), as.numeric))#making all variables numeric to use ML 

#reduced tbl contains variables about how income realtes to perceptions of others, beliefs about life + race
#made from the cleaned tbl to check if variables had enough information in them
gss_reduced_tbl <- gss_tbl %>%
  select(
    SEX,
    AGE, #didn't end up using age
    INCOME,# Family income
    FAIR, # Are people fair or try to take advantage
    TRUST,# Can people be trusted
    LIFE, #life exciting or dull?
    HAPPY, # Happiness level
    RACE #race of respondent (unfortunately just 2 options or other)
  )%>%
  mutate( #turning each variable into a factor per the codesheet
    fairness = factor(FAIR, levels = c(1, 2, 3), labels = c("Would take advantage", "Would try to be fair", "It depends")),
    trust=factor(TRUST, levels=c(1,2,3), labels = c("Most people can be trusted", "Can't be too careful", "Other/it depends")),
    life=factor(LIFE, levels=c(1,2,3), labels=c("Exciting", "Routine", "Dull")),
    happiness=factor(HAPPY, levels=c(1,2,3), labels=c("Very Happy", "Pretty happy", "Not too happy")),
    race=factor(RACE, levels=c(1,2,3), labels=c("White", "Black", "Other")),
    sex=factor(SEX, levels=c(1,2), labels=c("Male", "Female"))
  )%>%
  filter(!is.na(INCOME) & !is.na(fairness) & !is.na(trust) & !is.na(life) & !is.na(happiness) & !is.na(race) & !is.na(AGE) & !is.na(sex))

#saving the tbl to the shiny folder for the app
gss_reduced_tbl  %>%
  saveRDS("../shiny/import.RDS")     

# Visualization
#Creating a table of summary statistics for income
summary_table <- gss_tbl %>%
  summarise( #used this so I could easily emplpoy multiple functions together
    mean_income = mean(INCOME),
    median_income = median(INCOME),
    min_income = min(INCOME),
    max_income = max(INCOME)
  )
print(summary_table)

#creating a barplot of family income to display distribution of family income
(gss_tbl%>%
  ggplot( #using ggplot for the heightened control vs plot() base R
       aes(x=INCOME)) + 
  geom_bar(fill="green")+ #green like money!!
  labs(x="Total family income level (1-12)", #making axis labels
       y="Frequency",
       title="Total Family Income Level"))%>%
  ggsave("../figs/fig1.png",.,height=3,width=4, dpi=600) #saving the histogram

#R2: Are there significant differences in income level for different perceptions of fairness?
(gss_reduced_tbl%>%
    ggplot( #using ggplot for the heightened control vs plot() base R
      aes(x=INCOME, fairness)) + 
    geom_boxplot(fill="green")+ #green like money!!
    labs(x="Total family income level (1-12)", #making axis labels
         y="Fairness perception",
         title="Total Family Income Level vs Fairness perception"))%>%
  ggsave("../figs/fig2.png",.,height=4,width=6, dpi=600)

#R3: Are there significant differences in income level for different happiness levels?
(gss_reduced_tbl%>%
    ggplot( #using ggplot for the heightened control vs plot() base R
      aes(x=INCOME, happiness)) + 
    geom_boxplot(fill="green")+ #green like money!!
    labs(x="Total family income level (1-12)", #making axis labels
         y="Happiness level",
         title="Total Family Income Level vs Happiness level"))%>%
  ggsave("../figs/fig3.png",.,height=4,width=6, dpi=600)

#R4:Are there significant differences in income level for different races?
(gss_reduced_tbl%>%
    ggplot( #using ggplot for the heightened control vs plot() base R
      aes(x=INCOME, race)) + 
    geom_boxplot(fill="green")+ #green like money!!
    labs(x="Total family income level (1-12)", #making axis labels
         y="Race",
         title="Total Family Income Level vs Race"))%>%
  ggsave("../figs/fig4.png",.,height=4,width=6, dpi=600)

# Analysis

#R1: How effectively can the variables in the GSS 2018 dataset predict income?

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
#From this table, it seems that random forest was the best performing algorithm, as it was able to perform well on both training and novel data (a good balance of bias and variance). 
# It seems that income can be predicted pretty well from the GSS 2018 data as approximately 97% of the variation in income the variable being predicted) can be explained by the independent variable(s) in the regression model.
#The other models did not perform as well generally, as it seems some had difficulty performing on novel data (high bias), or just performed worse overall. 

#R2: Are there significant differences in income level for different perceptions of fairness?
aov_1<-aov(INCOME~fairness, data=gss_reduced_tbl) #running an anova considering income and fairness (similar to what was graphed in the app)
anova(aov_1) #printing anova result
(table_2<-apa.aov.table(aov_1, table.number=2))#creating a table for visualization using apaTables, because this is the easiest tool with good formatting -- compared to creating a table myself
TukeyHSD(aov_1) #printing Tukey's because results were significant (alpha = 0.05)
# Yes, there are significant differences in income level for different perceptions of fairness (alpha=0.05). A Tukey’s HSD test indicates that the respondents who answered “Would try to be fair” vs those who answered“Would try to take advantage” significantly differed by around 0.658 in level. Having more money seems to relate to perceiving others as more fair.

#R3: Are there significant differences in income level for different happiness levels?
aov_2<-aov(INCOME~happiness, data=gss_reduced_tbl) #also similar to what is graphed in the app, income x happiness anova
anova(aov_2)
(table_3<-apa.aov.table(aov_2, table.number=3)) #printing result table
TukeyHSD(aov_2) #printing Tukey's because results were significant
#There are significant differences in income level for different happiness levels(alpha=0.05). A Tukey’s HSD test indicates that the respondents who answered “Not too happy” vs  those who answered “Would try to take advantage” significantly differed by around -1.45 in level. Additionally, respondents who answered “Not too happy” vs “Pretty happy’ differed around -1.25 in level. Maybe money does buy happiness!

#R4:Are there significant differences in income level for different races?
aov_3<-aov(INCOME~race, data=gss_reduced_tbl)#also similar to what is graphed in the app, income x race anova
anova(aov_3)
(table_4<-apa.aov.table(aov_3, table.number=4))#printing result table
TukeyHSD(aov_3) #printing Tukey's because results were significant
#There are significant differences in income level for different races (alpha=0.05). A Tukey’s HSD test indicates that the respondents who identified as Black differed from those who identified as white by -0.93. It should be noted that there were only 3 categories for race in this dataset: white, black, and other.
