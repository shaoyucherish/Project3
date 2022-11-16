Project 3
================
Shaoyu Wang, Aniket Walimbe
2022-11-14

- <a href="#introduction" id="toc-introduction">Introduction</a>
- <a href="#required-packages" id="toc-required-packages">Required
  Packages</a>
- <a href="#data" id="toc-data">Data</a>
- <a href="#summarizations" id="toc-summarizations">Summarizations</a>
- <a href="#model" id="toc-model">Model</a>
- <a href="#comparison" id="toc-comparison">Comparison</a>
- <a href="#automation" id="toc-automation">Automation</a>

# Introduction

This [online news popularity data
set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity)
summarizes a heterogeneous set of features about articles published by
Mashable in a period of two years. There are 61 attributes, including 58
predictive attributes, 2 non-predictive, 1 goal field. The number of
shares is our target variable, and we select predictive variables from
the remaining variables based on the exploratory data analysis. The
purpose of our analysis is to predict the the number of shares. During
this project, we read and subset the data set at first, and split data
into training set and test set, then we create some basic summary
statistics and plots about the training data, at last we fit linear
regression models and ensemble tree-based models and test the
predictions.

# Required Packages

First, we need to load the required packages:

``` r
# Load libraries
library(readr)
library(tidyverse)
library(dplyr)
library(caret)
library(leaps)
library(ggplot2)
library(corrplot)
library(GGally)
library(randomForest)
```

# Data

Read in the data and subset the data to work on the data channel of
interest. We find that there are seven similar columns for weekdays from
Monday to Sunday, so we merge these columns and name the new variable as
`publish_weekday` and convert it to factor. For this step, we also
remove the non-predictive variables.

``` r
#Read in the data file
newsData <- read_csv("OnlineNewsPopularity.csv", show_col_types = FALSE)
#Choose the data channel of interest
if (params$channelID != "") {
  paramChannelName <- params$channelID
} else {
  paramChannelName <- "lifestyle"
}
channelID <- paste("data_channel_is_", paramChannelName, sep = "")
#Merge the weekday columns as one single column.
news <- newsData %>% 
  filter(.data[[channelID]] == 1) %>% 
  select(url, starts_with("weekday_is_")) %>% 
  pivot_longer(-url) %>% 
  filter(value != 0) %>% 
  mutate(publish_weekday = substr(name, 12, 20)) %>% 
  left_join(newsData, by = "url") %>% 
#Remove non predictive variables
  select(-c(url, name, value, timedelta, starts_with("data_channel_is_"), starts_with("weekday_is_")))
#convert publish_weekday to factor
news$publish_weekday <- as.factor(news$publish_weekday)
news
```

Split the data into a training set and a test set.

``` r
set.seed(111)
trainIndex <- createDataPartition(news$shares, p = 0.7, list = FALSE)
newsTrain <- news[trainIndex,]
newsTest <- news[-trainIndex,]
#newsTrain
```

# Summarizations

For this part, we produce some basic summary statistics and plots about
the training data.

- **Tables**

Firstly, here is a quick summary of all variables as shown below, so
that we can know the variables roughly.

``` r
summary(newsTrain)
```

    ##   publish_weekday n_tokens_title   n_tokens_content n_unique_tokens  n_non_stop_words n_non_stop_unique_tokens
    ##  friday   :208    Min.   : 3.000   Min.   :   0.0   Min.   :0.0000   Min.   :0.0000   Min.   :0.0000          
    ##  monday   :221    1st Qu.: 8.000   1st Qu.: 305.0   1st Qu.:0.4636   1st Qu.:1.0000   1st Qu.:0.6292          
    ##  saturday :133    Median :10.000   Median : 497.5   Median :0.5208   Median :1.0000   Median :0.6840          
    ##  sunday   :135    Mean   : 9.764   Mean   : 607.3   Mean   :0.5241   Mean   :0.9891   Mean   :0.6837          
    ##  thursday :254    3rd Qu.:11.000   3rd Qu.: 793.0   3rd Qu.:0.5899   3rd Qu.:1.0000   3rd Qu.:0.7521          
    ##  tuesday  :240    Max.   :17.000   Max.   :7413.0   Max.   :0.8248   Max.   :1.0000   Max.   :1.0000          
    ##  wednesday:281                                                                                                
    ##    num_hrefs      num_self_hrefs      num_imgs         num_videos      average_token_length  num_keywords   
    ##  Min.   :  0.00   Min.   : 0.000   Min.   :  0.000   Min.   : 0.0000   Min.   :0.000        Min.   : 3.000  
    ##  1st Qu.:  6.00   1st Qu.: 1.000   1st Qu.:  1.000   1st Qu.: 0.0000   1st Qu.:4.457        1st Qu.: 7.000  
    ##  Median : 10.00   Median : 2.000   Median :  1.000   Median : 0.0000   Median :4.621        Median : 9.000  
    ##  Mean   : 13.21   Mean   : 2.518   Mean   :  4.888   Mean   : 0.4572   Mean   :4.586        Mean   : 8.233  
    ##  3rd Qu.: 18.00   3rd Qu.: 3.000   3rd Qu.:  8.000   3rd Qu.: 0.0000   3rd Qu.:4.793        3rd Qu.:10.000  
    ##  Max.   :118.00   Max.   :27.000   Max.   :111.000   Max.   :50.0000   Max.   :5.749        Max.   :10.000  
    ##                                                                                                             
    ##    kw_min_min       kw_max_min      kw_avg_min        kw_min_max       kw_max_max       kw_avg_max    
    ##  Min.   : -1.00   Min.   :    0   Min.   :   -1.0   Min.   :     0   Min.   :     0   Min.   :     0  
    ##  1st Qu.: -1.00   1st Qu.:  488   1st Qu.:  184.2   1st Qu.:     0   1st Qu.:690400   1st Qu.:118659  
    ##  Median :  4.00   Median :  813   Median :  301.1   Median :     0   Median :843300   Median :181881  
    ##  Mean   : 41.45   Mean   : 1664   Mean   :  418.1   Mean   :  7217   Mean   :702035   Mean   :183400  
    ##  3rd Qu.:  4.00   3rd Qu.: 1300   3rd Qu.:  439.1   3rd Qu.:  6200   3rd Qu.:843300   3rd Qu.:248982  
    ##  Max.   :377.00   Max.   :98700   Max.   :14187.8   Max.   :208300   Max.   :843300   Max.   :491771  
    ##                                                                                                       
    ##    kw_min_avg     kw_max_avg      kw_avg_avg    self_reference_min_shares self_reference_max_shares
    ##  Min.   :   0   Min.   :    0   Min.   :    0   Min.   :     0            Min.   :     0.0         
    ##  1st Qu.:   0   1st Qu.: 4042   1st Qu.: 2642   1st Qu.:   624            1st Qu.:   965.2         
    ##  Median :   0   Median : 5036   Median : 3221   Median :  1700            Median :  2850.0         
    ##  Mean   :1054   Mean   : 6625   Mean   : 3404   Mean   :  4741            Mean   :  8053.0         
    ##  3rd Qu.:2274   3rd Qu.: 7166   3rd Qu.: 3926   3rd Qu.:  3800            3rd Qu.:  7225.0         
    ##  Max.   :3610   Max.   :98700   Max.   :20378   Max.   :144900            Max.   :690400.0         
    ##                                                                                                    
    ##  self_reference_avg_sharess   is_weekend         LDA_00            LDA_01            LDA_02       
    ##  Min.   :     0.0           Min.   :0.0000   Min.   :0.01818   Min.   :0.01819   Min.   :0.01819  
    ##  1st Qu.:   942.5           1st Qu.:0.0000   1st Qu.:0.02251   1st Qu.:0.02222   1st Qu.:0.02223  
    ##  Median :  2500.0           Median :0.0000   Median :0.02914   Median :0.02507   Median :0.02792  
    ##  Mean   :  6168.2           Mean   :0.1821   Mean   :0.17903   Mean   :0.06506   Mean   :0.08074  
    ##  3rd Qu.:  5625.0           3rd Qu.:0.0000   3rd Qu.:0.25518   3rd Qu.:0.04001   3rd Qu.:0.11889  
    ##  Max.   :401450.0           Max.   :1.0000   Max.   :0.91980   Max.   :0.62253   Max.   :0.67623  
    ##                                                                                                   
    ##      LDA_03            LDA_04        global_subjectivity global_sentiment_polarity global_rate_positive_words
    ##  Min.   :0.01820   Min.   :0.02014   Min.   :0.0000      Min.   :-0.37271          Min.   :0.00000           
    ##  1st Qu.:0.02249   1st Qu.:0.31664   1st Qu.:0.4263      1st Qu.: 0.09868          1st Qu.:0.03464           
    ##  Median :0.03043   Median :0.57032   Median :0.4762      Median : 0.14874          Median :0.04348           
    ##  Mean   :0.14444   Mean   :0.53073   Mean   :0.4736      Mean   : 0.15064          Mean   :0.04419           
    ##  3rd Qu.:0.21061   3rd Qu.:0.79919   3rd Qu.:0.5248      3rd Qu.: 0.20520          3rd Qu.:0.05296           
    ##  Max.   :0.91837   Max.   :0.91995   Max.   :0.8667      Max.   : 0.51389          Max.   :0.12139           
    ##                                                                                                              
    ##  global_rate_negative_words rate_positive_words rate_negative_words avg_positive_polarity
    ##  Min.   :0.00000            Min.   :0.0000      Min.   :0.0000      Min.   :0.0000       
    ##  1st Qu.:0.01050            1st Qu.:0.6632      1st Qu.:0.1852      1st Qu.:0.3359       
    ##  Median :0.01532            Median :0.7377      Median :0.2586      Median :0.3832       
    ##  Mean   :0.01626            Mean   :0.7214      Mean   :0.2677      Mean   :0.3828       
    ##  3rd Qu.:0.02085            3rd Qu.:0.8112      3rd Qu.:0.3333      3rd Qu.:0.4343       
    ##  Max.   :0.06180            Max.   :1.0000      Max.   :1.0000      Max.   :0.7553       
    ##                                                                                          
    ##  min_positive_polarity max_positive_polarity avg_negative_polarity min_negative_polarity max_negative_polarity
    ##  Min.   :0.00000       Min.   :0.00          Min.   :-1.0000       Min.   :-1.0000       Min.   :-1.000       
    ##  1st Qu.:0.05000       1st Qu.:0.70          1st Qu.:-0.3232       1st Qu.:-0.7143       1st Qu.:-0.125       
    ##  Median :0.10000       Median :0.90          Median :-0.2612       Median :-0.5000       Median :-0.100       
    ##  Mean   :0.09355       Mean   :0.83          Mean   :-0.2671       Mean   :-0.5566       Mean   :-0.105       
    ##  3rd Qu.:0.10000       3rd Qu.:1.00          3rd Qu.:-0.2033       3rd Qu.:-0.4000       3rd Qu.:-0.050       
    ##  Max.   :0.50000       Max.   :1.00          Max.   : 0.0000       Max.   : 0.0000       Max.   : 0.000       
    ##                                                                                                               
    ##  title_subjectivity title_sentiment_polarity abs_title_subjectivity abs_title_sentiment_polarity
    ##  Min.   :0.0000     Min.   :-1.0000          Min.   :0.0000         Min.   :0.0000              
    ##  1st Qu.:0.0000     1st Qu.: 0.0000          1st Qu.:0.2000         1st Qu.:0.0000              
    ##  Median :0.1000     Median : 0.0000          Median :0.5000         Median :0.0000              
    ##  Mean   :0.2827     Mean   : 0.1052          Mean   :0.3531         Mean   :0.1688              
    ##  3rd Qu.:0.5000     3rd Qu.: 0.2000          3rd Qu.:0.5000         3rd Qu.:0.2927              
    ##  Max.   :1.0000     Max.   : 1.0000          Max.   :0.5000         Max.   :1.0000              
    ##                                                                                                 
    ##      shares      
    ##  Min.   :    28  
    ##  1st Qu.:  1100  
    ##  Median :  1700  
    ##  Mean   :  3847  
    ##  3rd Qu.:  3225  
    ##  Max.   :208300  
    ## 

Then we can check our response variable `shares`. The below table shows
that the mean, standard deviation, median, IQR of `shares`.

``` r
#numerical summary for the variable shares
newsTrain %>% 
  summarise(mean = round(mean(shares), 0), sd = round(sd(shares), 0), 
            median = round(median(shares), 0), IQR = round(IQR(shares), 0))
```

We also obtain the numerical summaries on some subgroups. We choose four
example subgroups: number of images, number of videos, and number of
keywords, since people may concern more on these when they do searching
and sharing.

``` r
#numerical summaries on subgroups
newsTrain %>% 
  group_by(num_imgs) %>% 
  summarise(mean = round(mean(shares), 0), sd = round(sd(shares), 0), 
            median = round(median(shares), 0), IQR = round(IQR(shares), 0))

newsTrain %>% 
  group_by(num_videos) %>% 
  summarise(mean = round(mean(shares), 0), sd = round(sd(shares), 0), 
            median = round(median(shares), 0), IQR = round(IQR(shares), 0))

newsTrain %>% 
  group_by(num_keywords) %>% 
  summarise(mean = round(mean(shares), 0), sd = round(sd(shares), 0), 
            median = round(median(shares), 0), IQR = round(IQR(shares), 0))
```

Moreover, we divide the title subjectivity into 3 categories:  
1. High: greater than 0.8  
2. Medium: 0.4 to less than 0.8  
3. Low: less than 0.4  
The contingency table is then shown below.

``` r
newsTrain$subject_activity_type <- ifelse(newsTrain$title_subjectivity >= 0.8, "High", 
                                          ifelse(newsTrain$title_subjectivity >= 0.4, "Medium",
                                                 ifelse(airquality$Wind >= 0, "Low")))
table(newsTrain$subject_activity_type)
```

    ## 
    ##   High    Low Medium 
    ##    161    930    381

- **Plots**

At the beginning, let’s plot the correlation between the numeric
variables.

``` r
newsTrainsub <- newsTrain %>% select(-c(publish_weekday, subject_activity_type))
correlation <- cor(newsTrainsub, method = "spearman")
corrplot(correlation, tl.col = "black", tl.cex = 0.5)
```

![](README_files/figure-gfm/unnamed-chunk-72-1.png)<!-- --> From the
correlation graph above, we can see that the following variables seem to
be moderately correlated: - `n_tokens_contents`, `n_unique_tokens`,
`n_non_stop_words`, `n_non_stop_unique_tokens`, `num_hrefs`,
`num_imgs` - `kw_min_min`, `kw_max_min`, `kw_avg_min`, `kw_min_max`,
`kw_max_max`, `kw_avg_max`, `kw_min_avg`, `kw_max_avg`, `kw_avg_avg` -
`self_reference_min_shares`, `self_reference_max_shares`,
`self_reference_avg_sharess` - `LDA_00`, `LDA_01`, `LDA_02`, `LDA_03` -
`global_sentiment_polarity`, `global_rate_positive_words`,
`global_rate_negative_words`, `rate_positive_words`,
`rate_negative_words` - `avg_positive_polarity`,
`min_positive_polarity`, `max_positive_polarity` -
`avg_negative_polarity`, `min_negative_polarity`,
`max_negative_polarity` - `title_subjectivity`,
`title_sentiment_polarity`, `abs_title_subjectivity`,
`abs_title_sentiment_polarity`

For further EDA, we are going to plot graphs to see trends between
different variables with respect to the number of shares.

A plot between number of shares and article published day: This plot
shows the number of shares an article has based on the day that has been
published.

``` r
newsTrainday <- newsTrain %>%
  select(publish_weekday, shares) %>%
  group_by(publish_weekday) %>% 
  summarise(total_shares=sum(shares))

g <- ggplot(data = newsTrainday, aes(x=publish_weekday, y=total_shares))
g + geom_col(fill = "lightblue", color = "black") +
  labs(title = " Shares for articles published based on weekdays")
```

![](README_files/figure-gfm/unnamed-chunk-73-1.png)<!-- -->

Let’s select some variables as example to plot scatter plots.

A scatter plot with the number of shares on the y-axis and the number of
words in the title on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = n_tokens_title, y = shares))
g + geom_point() +
  labs(x = "Number of words in the title", y = "Number of shares", 
       title = "Scatter Plot: Number of words in the title VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-74-1.png)<!-- --> We can
inspect the trend of shares as a function of the number of words in the
title. Therefore, we can see that the number of words in title has an
effect on the number of shares.

A scatter plot with the number of shares on the y-axis and the number of
words in the content on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = n_tokens_content, y = shares))
g + geom_point() +
  labs(x = "Number of words in the content", y = "Number of shares", 
       title = "Scatter Plot: Number of words in the content VS Number of shares") 
```

![](README_files/figure-gfm/unnamed-chunk-75-1.png)<!-- --> From the
plot above, we can easily see that the number of shares is decreasing
while the the number of words in the content is increasing. So it can be
illustrated that the number of words in the content will affect the
number of shares.

A scatter plot with the number of shares on the y-axis and the number of
links to other articles published by Mashable on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = num_self_hrefs, y = shares))
g + geom_point() +
  labs(x = "Number of links to other articles published by Mashable", y = "Number of shares", 
       title = "Scatter Plot: Number of links to other articles published by Mashable VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-76-1.png)<!-- --> The plot
above shows that as the number of links to other articles increasing,
the number of shares is decreasing. So the the number of links to other
articles has an infulence on the number of shares.

A scatter plot with the number of shares on the y-axis and the number of
images on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = num_imgs, y = shares))
g + geom_point() +
  labs(x = "Number of images", y = "Number of shares", 
       title = "Scatter Plot: Number of images VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-77-1.png)<!-- --> The plot
above shows that the number of shares decreases as the number of images
increasing. Therefore, the number of images will affect the number of
shares as well.

A scatter plot with the number of shares on the y-axis and the number of
videos on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = num_videos, y = shares))
g + geom_point() +
  labs(x = "Number of videos", y = "Number of shares", 
       title = "Scatter Plot: Number of videos VS Number of shares") 
```

![](README_files/figure-gfm/unnamed-chunk-78-1.png)<!-- -->

A scatter plot with the number of shares on the y-axis and the average
length of words in content on the x-axis is created:

``` r
g <- ggplot(newsTrain, aes(x = average_token_length, y = shares))
g + geom_point() + 
  labs(x = "Average token length", y = "Number of shares", 
       title = "Scatter Plot: Average token length VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-79-1.png)<!-- --> Through the
plot above, we can see that the most of shares are between 4 and 6
words. The average token length will also affect the number of shares.

A scatter plot with the number of shares on the y-axis and the number of
keywords in the metadata on the x-axis is created:

``` r
g <- ggplot(newsTrain, aes(x = num_keywords, y = shares))
g + geom_point() + 
  labs(x = "Number of keywords in the metadata", y = "Number of shares", 
       title = "Scatter Plot: Number of keywords in the metadata VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-80-1.png)<!-- --> According to
the plot above, we can find that as the number of keywords increasing,
the number of shares is increasing. So the number of keywords in the
metadata will influence the number of shares.

A scatter plot with the number of shares on the y-axis and the text
subjectivity on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = global_subjectivity, y = shares))
g + geom_point() + 
  labs(x = "Text subjectivity", y = "Number of shares", 
       title = "Scatter Plot: Text subjectivity VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-81-1.png)<!-- --> From the
plot above, it presents that the most of shares are between 0.25 and
0.75 text subjectivity. So the text subjectivity will influence the
number of shares as well.

A scatter plot with the number of shares on the y-axis and the title
subjectivity on the x-axis is created:

``` r
g <- ggplot(data = newsTrain, aes(x = title_subjectivity, y = shares))
g + geom_point() + 
  labs(x = "Title subjectivity", y = "Number of shares", 
       title = "Scatter Plot: Title subjectivity VS Number of shares")
```

![](README_files/figure-gfm/unnamed-chunk-82-1.png)<!-- --> The plot
above shows that the title subjectivity has less effect on the number of
shares.

Through the analysis above, we will select predictors as follows: -
`publish_weekday`: The article published day  
- `n_tokens_title`: Number of words in the title  
- `n_tokens_content`: Number of words in the content  
- `num_self_hrefs`: Number of links to other articles published by
Mashable  
- `num_imgs`: Number of images  
- `num_videos`: Number of videos  
- `average_token_length`: Average length of the words in the content  
- `num_keywords`: Number of keywords in the metadata  
- `kw_avg_avg`: Avg. keyword (avg. shares)  
- `self_reference_avg_sharess`: Avg. shares of referenced articles in
Mashable  
- `LDA_04`: Closeness to LDA topic 4  
- `global_subjectivity`: Text subjectivity  
- `global_sentiment_polarity`: Text sentiment polarity  
- `avg_positive_polarity`: Avg. polarity of positive words -
`avg_negative_polarity`: Avg. polarity of negative words

``` r
set.seed(111)
Train <- newsTrain %>% 
  select(publish_weekday, n_tokens_title, n_tokens_content, num_self_hrefs, num_imgs, num_videos, average_token_length, num_keywords, kw_avg_avg, self_reference_avg_sharess, LDA_04, global_subjectivity, global_sentiment_polarity, avg_positive_polarity, avg_negative_polarity, shares)
Test <- newsTest %>% 
  select(publish_weekday, n_tokens_title, n_tokens_content, num_self_hrefs, num_imgs, num_videos, average_token_length, num_keywords, kw_avg_avg, self_reference_avg_sharess, LDA_04, global_subjectivity, global_sentiment_polarity, avg_positive_polarity, avg_negative_polarity, shares)
#Train
```

# Model

- **Linear Regression Model**

First, we fit a forward stepwise linear regression model for the
training dataset. The data is centered and scaled and number of shares
is the response variable.

``` r
#forward stepwise
set.seed(111)
fwFit <- train(shares ~ ., data = Train,
               method = "leapForward",
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 5))
#summary(fwFit)
fwFit
```

    ## Linear Regression with Forward Selection 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## Pre-processing: centered (20), scaled (20) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1176, 1178, 1178, 1178, 1178 
    ## Resampling results across tuning parameters:
    ## 
    ##   nvmax  RMSE      Rsquared     MAE     
    ##   2      8936.405  0.007010645  3566.950
    ##   3      8922.556  0.009613257  3553.279
    ##   4      8937.825  0.008357905  3559.836
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was nvmax = 3.

We also fit a backward stepwise linear regression model for the training
dataset. The data is centered and scaled and number of shares is the
response variable.

``` r
#backward stepwise
set.seed(111)
bwFit <- train(shares ~ ., data = Train,
               method = "leapBackward",
               preProcess = c("center", "scale"),
               trControl = trainControl(method = "cv", number = 5))
#summary(bwFit)
bwFit
```

    ## Linear Regression with Backwards Selection 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## Pre-processing: centered (20), scaled (20) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1176, 1178, 1178, 1178, 1178 
    ## Resampling results across tuning parameters:
    ## 
    ##   nvmax  RMSE      Rsquared     MAE     
    ##   2      8942.405  0.003838792  3578.035
    ##   3      8922.556  0.009613257  3553.279
    ##   4      8948.032  0.006784268  3565.545
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was nvmax = 3.

Then we fit a linear regression model with all predictors.

``` r
#with all predictors
set.seed(111)
lrFit <- train(shares ~ ., data = Train,
               method = "lm",
               trControl = trainControl(method = "cv", number = 5))
lrFit
```

    ## Linear Regression 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1176, 1178, 1178, 1178, 1178 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared     MAE    
    ##   8988.258  0.005722325  3646.68
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE

- **Random Forest Model**

Next, we have fitted a random forest model which is an example of an
ensemble based-tree model. Instead of traditional decision trees, a
random forest tree will take a random subset of the predictors for each
tree fit and calculate the average of results.

``` r
set.seed(111)
randomFit <- train(shares ~ ., 
                   data = Train, 
                   method = "rf",
                   preProcess = c("center","scale"),
                   trControl = trainControl(method = "cv", number = 5),
                   tuneGrid = data.frame(mtry = ncol(Train)/3))
randomFit
```

    ## Random Forest 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## Pre-processing: centered (20), scaled (20) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1176, 1178, 1178, 1178, 1178 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared    MAE   
    ##   9115.357  0.01349964  3780.3
    ## 
    ## Tuning parameter 'mtry' was held constant at a value of 5.333333

- **Boosted Tree Model**

Moreover, we have fitted a boosted tree model which is another ensemble
based-tree model. Boosted tree models are combination of two techniques:
decision tree algorithms and boosting methods. It repeatedly fits many
decision trees to improve the accuracy of the model.

``` r
set.seed(111)
boostedFit <- train(shares ~ ., 
                    data = Train, 
                    method = "gbm", 
                    preProcess = c("center", "scale"),
                    trControl = trainControl(method = "cv", number = 5),
                    tuneGrid = expand.grid(n.trees = c(25,50,100,150,200), 
                                           interaction.depth = c(1,2,3,4), 
                                           shrinkage = 0.1, 
                                           n.minobsinnode = 10),
                    verbose = FALSE)
boostedFit
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 1472 samples
    ##   15 predictor
    ## 
    ## Pre-processing: centered (20), scaled (20) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 1176, 1178, 1178, 1178, 1178 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   25      8995.960  0.003774617  3585.314
    ##   1                   50      9034.606  0.005083773  3593.280
    ##   1                  100      9073.458  0.005429124  3599.658
    ##   1                  150      9155.535  0.005817416  3641.084
    ##   1                  200      9175.506  0.004024536  3665.518
    ##   2                   25      9077.295  0.004924311  3620.646
    ##   2                   50      9174.961  0.006614410  3671.884
    ##   2                  100      9273.046  0.006044017  3778.624
    ##   2                  150      9319.711  0.006403743  3798.421
    ##   2                  200      9386.835  0.009301048  3856.783
    ##   3                   25      9096.839  0.007953789  3647.894
    ##   3                   50      9213.110  0.008197227  3708.845
    ##   3                  100      9366.736  0.010336261  3843.707
    ##   3                  150      9368.423  0.012143219  3889.817
    ##   3                  200      9396.693  0.012818471  3924.872
    ##   4                   25      9089.313  0.010528036  3618.082
    ##   4                   50      9120.178  0.016334623  3660.944
    ##   4                  100      9251.464  0.018178769  3779.965
    ##   4                  150      9299.776  0.019009071  3817.132
    ##   4                  200      9389.695  0.020166334  3881.671
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode' was
    ##  held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 1, shrinkage = 0.1
    ##  and n.minobsinnode = 10.

# Comparison

All the models are compared by RMSE on the test set.

``` r
#fit a linear regression model
fw_mod <- postResample(predict(fwFit, newdata = Test), obs = Test$shares)
bw_mod <- postResample(predict(bwFit, newdata = Test), obs = Test$shares)
lr_mod <- postResample(predict(lrFit, newdata = Test), obs = Test$shares)
#random forest
random_mod <- postResample(predict(randomFit, newdata = Test), obs = Test$shares)
#boosted tree
boosted_mod <- postResample(predict(boostedFit, newdata = Test), obs = Test$shares)
#compare all models
tibble(model = c("Forward",
                 "Backward",
                 "LR with all predictors",
                 "Random Forest",
                 "Boosted Tree"), 
       RMSE = c(fw_mod[1],
                bw_mod[1],
                lr_mod[1],
                random_mod[1],
                boosted_mod[1]))
```

From the above table, we can find that forward and backward stepwise
models have the same RMSE value. So we can say that, among these models,
the best model is forward or backward stepwise model since it has the
least RMSE value as compared to the other models.

# Automation

For this automation part, we want to produce the similar reports for
each news channels. We firstly create a set of parameters, which match
with 6 channels. Then read the parameter and subset the data with the
specified channel. After everything is ready, run the below chunk of
code in the console, we will automatically get the reports for each news
channel.

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
