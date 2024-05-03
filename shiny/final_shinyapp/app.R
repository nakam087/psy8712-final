library(shiny)
library(ggplot2)
library(dplyr)

import_tbl<-readRDS("import.RDS")

# Define UI for application that draws a histogram of income levels
ui <- fluidPage(
    # Application title
    titlePanel("How does income relate to perceptions of life and others?"),
    # Sidebar with radio buttons with 6 choices: filtering based on sex, happiness, race, trust level, perception of fairness, life excitement
    sidebarLayout(
        sidebarPanel( #on the left side as a panel of radio buttons, chose radio buttons because it looks like a multiple choice with my factor options
            radioButtons("sex_select", #naming so can work with this below
                         label="Select Sex", #What appears to viewers of the app
                         choices=c("Male", "Female", "All"), #options that appear on the buttons
                         selected="All"), #default on all of the radio buttons is the most inclusive
            radioButtons("happy_select", #choose happiness level
                         label="Select happiness level",
                         choices=c("Very Happy", "Pretty happy", "Not too happy", "All"),
                         selected="All"),
            radioButtons("race_select", #choose race
                         label="Select race",
                         choices=c("White", "Black", "Other", "All"),
                         selected="All"),
            radioButtons("trust_select", #choose trust in others
                         label="Select trust level in others",
                         choices=c("Most people can be trusted", "Can't be too careful", "Other/it depends", "All"),
                         selected="All"),
            radioButtons("fair_select", #choose perceptions of fairness
                         label="Select perception of fairness",
                         choices=c("Would take advantage", "Would try to be fair", "It depends", "All"),
                         selected="All"),
            radioButtons("life_select", #choose life perception
                         label="Select level of life excitement",
                         choices=c("Exciting", "Routine", "Dull", "All"),
                         selected="All")
            
        ),

        # Show a plot of the generated distribution
        mainPanel(
           plotOutput("distPlot") #the main section of the app will contain a histogram plot
        )
    )
)

# Define server logic required to draw a histogram of income level
server <- function(input, output) {
    output$distPlot <- renderPlot({ #the following things will change how the plot appears
      filtered_tbl <- import_tbl #creating new dataset to filter based on the inputs above
      
      # Apply filters successively
      filtered_tbl <- filtered_tbl %>% # i know i am not supposed to do this, but for some reason, my code was failing without it
        filter(if (input$sex_select != "All") sex == input$sex_select else TRUE) %>% #if an option other than nothing is selected, the tbl will be filtered to male or female, depending. This is also true for the following options, as the categories are already made into the same factors that I have in the ui section
        filter(if (input$happy_select != "All") happiness == input$happy_select else TRUE) %>%
        filter(if (input$race_select != "All") race == input$race_select else TRUE) %>%
        filter(if (input$trust_select != "All") trust == input$trust_select else TRUE) %>%
        filter(if (input$fair_select != "All") fairness == input$fair_select else TRUE) %>%
        filter(if (input$life_select != "All") life == input$life_select else TRUE) 
       
        # draw the histogram
        ggplot(filtered_tbl,
               aes(x=INCOME)) +
          geom_histogram(fill="green") +
          labs(x="Income level 1-12",
               title = "Family Income level from GSS 2018 data")
    })
}
# Run the application 
shinyApp(ui = ui, server = server)
