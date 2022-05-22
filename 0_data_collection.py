# Retrieve the tweets containing key words within certain period

# import libraries
import twint
import pandas as pd
from datetime import timedelta

def tweet_retriever(name, since, until):
    c = twint.Config()
    c.Lang = "en"        # language = English
    c.Search = name      # containing key words (name of athletes)
    c.Pandas = True      # format of returned tweets
    c.Limit = 50         # number of tweets
    c.Since = since      # start date
    c.Until = until      # end date
    twint.run.Search(c)
    return(twint.storage.panda.Tweets_df)

athletes = pd.read_csv('directory_to_your_file')  # a file containing all athletes' names

# search the name of athletes during certain period on Twitter
daterange = pd.date_range('2021-08-10', '2022-01-30')

output = []   # initialize output

# itearte on all names, get tweets
for name in athletes.name:
    for start_date in daterange:
        since = start_date.strftime("%Y-%m-%d")
        until = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")
        tweets = tweet_retriever(name, since, until)
        output.append(tweets)

output_df = pd.concat(output)

# save to a csv file
output_df.to_csv('directory_to_put_your_file')
