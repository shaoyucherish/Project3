Shaoyu Wang, Aniket Walimbe
2022-11-14


# Description
This repo was created by Shaoyu Wang and Aniket Walimbe as a part of Project 3 for the ST 558 - Data Science for Statisticians course. The project involves creating predictive models and automating Markdown reports. The data set we analyzed for this project is [online news popularity data set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity), which summarizes a heterogeneous set of features about articles published by Mashable in a period of two years. We fitted regression models and ensemble tree-based models and selected the best model for predicting the response variable. The creation of analysis reports were automated for 6 different news channels.

# A list of R packages used

- readr
- tidyverse
- dplyr
- caret
- leaps
- ggplot2
- corrplot
- randomForest
- rmarkdown

# Links to the analyses

- The analysis for [Lifestyle](https://shaoyucherish.github.io/Project3/lifestyle.html)  
- The analysis for [Entertainment](https://shaoyucherish.github.io/Project3/entertainment.html)  
- The analysis for [Business](https://shaoyucherish.github.io/Project3/bus.html)  
- The analysis for [Social Media](https://shaoyucherish.github.io/Project3/socmed.html)  
- The analysis for [Tech](https://shaoyucherish.github.io/Project3/tech.html)  
- The analysis for [World](https://shaoyucherish.github.io/Project3/world.html)  

# Automatically generate analysis reports

``` r
#create channel names
channelID <- data.frame("lifestyle", "entertainment", "bus", "socmed", "tech", "world")
#create filenames
output_file <- paste0(channelID,".md")
#create a list for each channel with the channel name parameter
params = lapply(channelID, FUN = function(x){list(channelID = x)})
#put into a data frame
reports <- tibble(output_file, params)
#render code
apply(reports, MARGIN = 1,
          FUN = function(x){
             rmarkdown::render(input = "project3.Rmd",
             output_format = "github_document",
             output_file = x[[1]],
             params = x[[2]])
             })
```
