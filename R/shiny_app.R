# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
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


