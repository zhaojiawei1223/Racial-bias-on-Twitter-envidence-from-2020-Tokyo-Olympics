# News coverage analysis is performed for sports media accounts.
# We will compare the coverage ratio of atheltes from different races with their proportion on the roster.
# For example, if there are 20% black athletes on the roster, whereas only 10% news covering them,
# then we say there is racial bias towards black athletes.
# To deal with the "pop star effects", we also do the analysis above when not including the top 5 stars.
# After this, the most distinctive words were found for the news covering the minority and white athletes.

# import libraries
import pandas as pd
import preprocessor as p  # tweet-preprocessor for preprocessing tweets
from collections import Counter
from scipy.stats import chi2_contingency
import spacy
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# read tweets and roster
tweets = pd.read_json('your_file')  # two columns: tweets and media
athletes = pd.read_csv('your_file')  # two columns: name and racial

# preprocess tweets
tweets['tweets'] = tweets['tweets'].apply(lambda x: p.clean(x))

# get the number of white athletes and minority athletes in roster
athletes.racial.value_counts()  # white: 491 (78.94%), minority: 131 (21.06%)

# find the number of minority athletes showing in news
athletes_minority = athletes[athletes.racial != 'white']
no_minority = 0
for name in athletes_minority.name:
  for tweet in tweets.tweets:
    if name in tweet:
      no_minority += 1
print(no_minority)  # 176 people (21.70%)

# the same was done for white athletes
athletes_white = athletes[athletes.racial == 'white']
no_whilte = 0
for name in athletes_white.name:
  for tweet in tweets.tweets:
    if name in tweet:
      no_whilte += 1
print(no_whilte)  # 635 people (78.30%)

# add a label to tweets belonging to minority athletes
# (since a tweet may cover several athletes, the number of news and the number of athletes showing in news may differ)
match = []
for tweet in tweets.tweets:
  match_number = 0
  for name in athletes_minority.name:
    if name in tweet:
      match_number += 1
  if match_number != 0:
    match.append('minority')
  else:
    match.append('none')
tweets['match'] = match
tweets_minority = tweets[tweets.match == 'minority']  # 145 tweets (25.94%)

# the same was done to white athletes
match = []
for tweet in tweets.tweets:
  match_number = 0
  for name in athletes_white.name:
    if name in tweet:
        match_number += 1
  if match_number != 0:
      match.append('white')
  else:
    match.append('none')
tweets['match'] = match
tweets_white = tweets[tweets.match == 'white']  # 414 tweets (74.06%)

# To deal with the "pop star effects", do the analysis above when not including the top 5 stars.
tweets_white.value_counts().head() # this shows the top5 stars among white athelets based on coverage ratio
# remove those names in athletes_white.name and do the coverage ratio analysis above for different groups


# distinctive words in news covering minority athletes and white athletes
# tokenize tweets_minority
nlp = spacy.load("en_core_web_sm")
processed_texts = [text for text in nlp.pipe(tweets_minority.tweets, disable=["ner","parser"])]
tokenized_texts = [[word.lemma_.lower() for word in processed_text if not word.is_stop and not word.is_punct and word.pos_ == 'ADJ']
                    for processed_text in processed_texts]
tweets_minority['tokenized_tweets'] = tokenized_texts

# tokenize tweets_white
processed_texts = [text for text in nlp.pipe(tweets_white.tweets, disable=["ner","parser"])]
tokenized_texts = [[word.lemma_.lower() for word in processed_text if not word.is_stop and not word.is_punct and word.pos_ == 'ADJ']
                    for processed_text in processed_texts]
tweets_white['tokenized_tweets'] = tokenized_texts

# a function to flatten corpus
flatten = lambda t: [item for sublist in t for item in sublist]

def distinctive_words(target_corpus, reference_corpus):
    '''
    A funtion to find the most distinctive words of target corpus compared with reference corpus
    Args:
        target_corpus
        reference_corpus
    Returns:
        A data frame of distinctive words
    '''
    counts_c1 = Counter(target_corpus)   # don't forget to flatten your texts!
    counts_c2 = Counter(reference_corpus)
    vocabulary = set(list(counts_c1.keys()) + list(counts_c2.keys()))
    freq_c1_total = sum(counts_c1.values())
    freq_c2_total = sum(counts_c2.values())
    results = []
    for word in vocabulary:
        freq_c1 = counts_c1[word]
        freq_c2 = counts_c2[word]
        freq_c1_other = freq_c1_total - freq_c1
        freq_c2_other = freq_c2_total - freq_c2
        llr, p_value,_,_ = chi2_contingency([[freq_c1, freq_c2], [freq_c1_other, freq_c2_other]], lambda_='log-likelihood')
        if freq_c2 / freq_c2_other > freq_c1 / freq_c1_other:
            llr = -llr
        result = {'word':word, 'llr':llr,  'p_value': p_value}
        results.append(result)
    results_df = pd.DataFrame(results)
    return results_df

# find the distinctive words of minority athletes
distinctive_words_minority = distinctive_words(flatten(tweets_minority.tokenized_tweets),
                                               flatten(tweets_white.tokenized_tweets))
distinctive_words_minority = pd.DataFrame(distinctive_words_minority)
distinctive_words_minority.sort_values('llr', ascending=False)

# the same was done for white athletes
distinctive_words_white = distinctive_words(flatten(tweets_minority.tokenized_tweets),
                                            flatten(tweets_white.tokenized_tweets))
distinctive_words_white = pd.DataFrame(distinctive_words_white )
distinctive_words_white .sort_values('llr', ascending=False)

# use world cloud to show the results above
wc_minority_during = WordCloud(background_color="white", mask= mask, min_font_size = 10).generate('file_of_your_words')
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wc_minority_during)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

