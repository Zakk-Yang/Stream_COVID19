import os
import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



customer_key = os.environ['customer_key']
customer_secret = os.environ['customer_secret']
access_token = os.environ['access_token']
access_secret = os.environ['access_secret']

auth = tweepy.OAuthHandler(customer_key, customer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=False)

def stream_tweets(search_term, location, retreive_num):
    '''
    A function that returns a pandas dataframe for the keywords

    '''
    data = []  # empty list to which tweet_details obj will be added
    counter = 0  # counter to keep track of each iteration
    places = api.geo_search(query=location, granularity="country")
    for x in places:
        if x.name == location:
            place_id = x.id

    for tweet in tweepy.Cursor(api.search,
                               q=('{} AND place:{}'.format(search_term, place_id)),
                               count=retreive_num, lang='en', tweet_mode='extended').items():
        tweet_details = {}
        tweet_details['name'] = tweet.user.screen_name
        tweet_details['tweet'] = tweet.full_text
        tweet_details['retweets'] = tweet.retweet_count
        tweet_details['location'] = tweet.user.location
        tweet_details['country'] = location
        tweet_details['created'] = tweet.created_at.strftime("%d-%b-%Y")
        tweet_details['followers'] = tweet.user.followers_count
        tweet_details['is_user_verified'] = tweet.user.verified
        data.append(tweet_details)

        counter += 1
        if counter == retreive_num:
            break
        else:
            pass
    return data


def get_world_sentiment_df(search_terms, retreive_num):
    location = 'United Kingdom'
    for search_term in search_terms:
        uk = pd.DataFrame(stream_tweets(search_term, location, retreive_num))

    location = 'United States'
    for search_term in search_terms:
        us = pd.DataFrame(stream_tweets(search_term, location, retreive_num))

    location = 'Brazil'
    for search_term in search_terms:
        brazil = pd.DataFrame(stream_tweets(search_term, location, retreive_num))

    location = 'Italy'
    for search_term in search_terms:
        italy = pd.DataFrame(stream_tweets(search_term, location, retreive_num))

    location = 'Spain'
    for search_term in search_terms:
        spain = pd.DataFrame(stream_tweets(search_term, location, retreive_num))

    location = 'France'
    for search_term in search_terms:
        france = pd.DataFrame(stream_tweets(search_term, location, retreive_num))

    df = pd.concat([uk, us, brazil, italy, spain, france], axis=0)
    return df


df = get_world_sentiment_df('covid', 100)


def vader_sentiment_calc(tweet):
    try:
        analyser = SentimentIntensityAnalyzer()
        analysis = analyser.polarity_scores(tweet)
        if analysis['neu'] == 1 :
            return  'neutral'
        elif analysis['pos'] <= 0.1 and analysis['pos'] - analysis['neg'] <= 0 and analysis['compound']<0 :
            return  'negative'
        else:
            return 'positive'
    except:
        return None

df['vader_sentiment'] = df['tweet'].apply(vader_sentiment_calc)

df.to_csv('sentiment_df.csv')

