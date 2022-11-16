Project 3
================
Shaoyu Wang, Aniket Walimbe
2022-11-14


# Description



# A list of R packages used

- readr
- tidyverse
- dplyr
- caret
- leaps
- ggplot2
- corrplot
- GGally
- randomForest

# Links to the genterated analyses

[Lifestyle]()
[Entertainment]()
[Business]()
[Social Media]()
[Tech]()
[World]()

# Automation

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
             params = x[[2]],
             output_options = list(toc=TRUE, toc_depth=1, toc_float=TRUE))
             })
```
