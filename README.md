Shaoyu Wang, Aniket Walimbe
2022-11-14


# Description
The purpose of this repo

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

- The analysis for [Lifestyle](https://github.com/shaoyucherish/Project3/blob/main/lifestyle.html)  
- The analysis for [Entertainment](https://github.com/shaoyucherish/Project3/blob/main/entertainment.html)  
- The analysis for [Business](https://github.com/shaoyucherish/Project3/blob/main/bus.html)  
- The analysis for [Social Media](https://github.com/shaoyucherish/Project3/blob/main/socmed.html)  
- The analysis for [Tech](https://github.com/shaoyucherish/Project3/blob/main/tech.html)  
- The analysis for [World](https://github.com/shaoyucherish/Project3/blob/main/world.html)  

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
             params = x[[2]],
             output_options = list(toc=TRUE, toc_depth=1, toc_float=TRUE))
             })
```
