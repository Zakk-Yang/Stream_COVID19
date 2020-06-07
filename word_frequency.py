import string
import nltk
from nltk.stem.porter import *
from collections import Counter
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
import pandas as pd
import streamlit as st
import numpy as np

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


def clean_text(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation and remove words containing numbers.'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    '''Get rid of some additional punctuation and non-sensical text that was missed the first time around.'''
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub(r'\b\w{1,3}\b', '', text) #remove any words that is less than 3
    # remove twitter Return handles (RT @xxx:)
    text = np.vectorize(remove_pattern)(text, "RT @[\w]*:")

    # remove twitter handles (@xxx)
    text = np.vectorize(remove_pattern)(text, "@[\w]*")

    # remove URL links (httpxxx)
    text = np.vectorize(remove_pattern)(text, "https?://[A-Za-z0-9./]*")

    # remove special characters, numbers, punctuations (except for #)
    text = np.core.defchararray.replace(text, "[^a-zA-Z]", " ")
    return text


def token_stem(text):
    text = text.split()  # tokenization
    stemmer = PorterStemmer()
    stemmed_text = [stemmer.stem(i) for i in text]
    return stemmed_text


def list_to_string(df, text_column):
    tokenized_tweet = df[text_column]
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    df[text_column] = tokenized_tweet
    return df[text_column]


def word_cloud(df, text_column, additional_stop_words=None):
    all_words = ' '.join([x for x in df[text_column]])
    add_stop_words = [word for word, count in Counter(all_words).most_common() if count > 100]
    add_stop_words.extend(additional_stop_words)
    stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)
    plt.figure(figsize=(10, 7))
    plt.axis('off')
    wordcloud = WordCloud(stopwords=stop_words, width=800,
                          height=500, random_state=21, max_font_size=110,
                          background_color='white',
                          ).generate(all_words)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.show()
    return st.pyplot()

# function to collect all hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
        # unnesting list
    hashtags = sum(hashtags, [])
    hashtags = [clean_text(a) for a in hashtags]
    hashtags = [string for string in hashtags if string != ""]
    return hashtags


def hash_tag_plot(df, text_column, exclude_list, title):
    tag_list = hashtag_extract(df[text_column])
    for x in tag_list:
        if x in exclude_list:
            tag_list.remove(x)
    a = nltk.FreqDist(tag_list)

    d = pd.DataFrame({'Hashtag': list(a.keys()),
                      'Count': list(a.values())})
    d.Hashtag = d.Hashtag.str.strip()
    d.dropna(how='any', inplace=True)
    d.sort_values(by='Count', ascending=False, inplace=True)
    # selecting top 10 most frequent hashtags
    d = d.nlargest(columns="Count", n=10)
    plt.figure(figsize = (20,5))
    sns.set(font_scale=1.1)
    ax = sns.barplot(data=d, x="Hashtag", y="Count")
    plt.title(title)
    ax.set(ylabel='Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=8)
    plt.tight_layout()
    plt.show()
    return st.pyplot()




