import os
import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sqlalchemy import create_engine


customer_key = os.environ['customer_key']
customer_secret = os.environ['customer_secret']
access_token = os.environ['access_token']
access_secret = os.environ['access_secret']
postgres_database_password = os.environ['postgres_password']




auth = tweepy.OAuthHandler(customer_key, customer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

search_term = 'covid'
retrieve_num = 800

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
                               q=('{} -filter:retweets AND place:{} '.format(search_term, place_id)),
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
    return pd.DataFrame(data)


location = 'United Kingdom'
uk = pd.DataFrame(stream_tweets(search_term, location, retrieve_num))

location = 'United States'
us = pd.DataFrame(stream_tweets(search_term, location, retrieve_num))

location = 'Brazil'
brazil = pd.DataFrame(stream_tweets(search_term, location, retrieve_num))

location = 'Italy'
italy = pd.DataFrame(stream_tweets(search_term, location, retrieve_num))

location = 'Spain'
spain = pd.DataFrame(stream_tweets(search_term, location, retrieve_num))

df = pd.concat([uk, us, brazil, italy, spain], axis=0)

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

# df.to_csv('sentiment_df.csv')
#
# def create_table():
#     try:
#         connection = psycopg2.connect(
#             user="erugboqiaqbruv",
#             password=postgres_database_password,
#             host="ec2-34-230-149-169.compute-1.amazonaws.com",
#             port="5432",
#             database="db72j9mubepavv"
#         )
#         c = connection.cursor()
#         c.execute("CREATE TABLE IF NOT EXISTS sentiment(country TEXT, tweet_date DATE NOT NULL, tweet TEXT, sentiment REAL)")
#         c.execute("CREATE INDEX fast_country ON sentiment(country)")
#         c.execute("CREATE INDEX fast_date ON sentiment(tweet_date)")
#         c.execute("CREATE INDEX fast_tweet ON sentiment(tweet)")
#         c.execute("CREATE INDEX fast_sentiment ON sentiment(sentiment)")
#         connection.commit()
#     except Exception as e:
#         print(str(e))
#
# create_table()



# df.to_sql('sentiment', con=engine)
URI = os.environ['URI']
engine = create_engine(URI)
df.to_sql('sentiment', con=engine, if_exists= 'replace')

