# Racial-bias-on-Twitter-envidence-from-2020-Tokyo-Olympics
This research focused on whether there is racial bias in sports on Twitter, and whether the Tokyo Olympics have an impact on this. Users on Twitter were divided into sports media and general users, analyzed seperately. Main methods used were news coverage analysis and sentiment analysis. 

## 0-Data-collection

To analyze news coverage of different races, we selected five top American sports media on Twitter, according to their followers and number of posts. The selected media are @espn, @Slonw, @FOXSports, @BleacherReport, and @TeamUSA. A total of 3917 tweets posted by them during and three months before the games (24 April 2021 to 8 August 2021) were crawled. 

For general users, we crawled all tweets that mentioned athletes’ names. The Python library we used was `Twint`. According to athletes’ races and periods of the posts, the dataset can be split into four groups – “white – before”, “minority – before”, “white – during”, and “minority – during”.

In order to get balanced data, we used the number of tweets for “minority – during” as a reference, and random sampling was conducted for the three other groups.

## 1-Racial-bias-analysis-on-sports-media
### 1.1  News coverage
The proportion of news coverage of white and minority athletes can be computed for each media account to quantitatively inspect whether there is racial bias. Based on the bar chart, there is no racial bias. The coverage ratios in the two periods are almost the same as the percentage in the roster.

<img width="725" alt="image" src="https://user-images.githubusercontent.com/105099474/169692040-92c76781-1613-4b1a-ab95-0056fdbb2d37.png">
<img width="682" alt="image" src="https://user-images.githubusercontent.com/105099474/169692086-9d5e9c7b-f139-4688-bae3-a179e78c4216.png">

In addition, we would like to discusse if the media coverage mainly focused on sports stars. We did a statistical analysis of athletes with top 5 media coverage. From the table below, the phenomenon of imbalanced coverage can be detected in both periods, especially in minority athletes. 

<img width="746" alt="image" src="https://user-images.githubusercontent.com/105099474/169692289-ac91af1b-d0cc-4880-9aa1-12e7b070dc51.png">

### 1.2 Distinctive words
We focused on finding the most distinctive words in media posts about white and minority athletes and inspecting their differences. The method to find the most distinctive words is log-likelihood ratio test. We concluded that no bias exists according to this, although slighly more positive words were detected from while athletes.

## 2-Racial-bias-analysis-on-general-users
In our study, sentiments are divided into positive, neutral, and negative. Our training set includes two parts – manually labeled tweets and publicly available Twitter sentiment corpus on Kaggle. We tried Logistic model, `BERT-base`, `BERT-weet` to get the classification model with highest accurracy. Then, used this model to label all other unlabeled tweets.

<img width="659" alt="image" src="https://user-images.githubusercontent.com/105099474/169692582-d4acf96e-f54b-45ec-bf4e-83421913a0df.png">

As a supplement, we collected tweets for different ethnical groups from 9th August until now to validate whether the negative sentiments returned to baseline, and inspect the progress brought by Tokyo Olympics is permanent or temporary. 

<img width="570" alt="image" src="https://user-images.githubusercontent.com/105099474/169692761-ecc0248c-4c42-4733-ad9d-fc066af488f1.png">


