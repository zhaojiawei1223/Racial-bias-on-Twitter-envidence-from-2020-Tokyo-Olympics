import twint
import pandas as pd
from datetime import timedelta

def tweet_retriever(name, since, until):
    c = twint.Config()
    c.Lang = "en"
    c.Search = name
    c.Pandas = True
    c.Limit = 50
    c.Since = since
    c.Until = until
    twint.run.Search(c)
    return(twint.storage.panda.Tweets_df)


# search the name of athletes during each period on Twitter
daterange = pd.date_range('2021-08-10', '2022-01-30')
output = []
athletes = pd.read_csv('directory_to_your_file')
for name in athletes.name:
    for start_date in daterange:
        since = start_date.strftime("%Y-%m-%d")
        until = (start_date + timedelta(days=1)).strftime("%Y-%m-%d")
        tweets = tweet_retriever(name, since, until)
        output.append(tweets)

output_df = pd.concat(output)
output_df.to_csv('directory_to_put_your_file')
