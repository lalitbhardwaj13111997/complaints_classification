
import pandas as pd
import numpy as np
import psycopg2 as pg
import pandas.io.sql as sql
import pandas as pd 
import numpy as np 
import preprocessor as p
import gensim
from sklearn.naive_bayes import GaussianNB
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re
from textblob import TextBlob
from  .models import sentiment



class sentiment1:
    def condition(self,x):
        if x<0:
            return "negative"
        elif x>0:
            return "positive"
        else:
            return "Neutral"        

  
    def getsentiment():
        query="""select * from tweets_analysis_tata_tweets"""
        conn=pg.connect(dbname="tweets",user="postgres",password="1234",host="localhost",port="5432")
        sw_list = stopwords.words('english')


        tweets_df=sql.read_sql_query(query,conn)
        tweets_df.drop(columns=['id'],inplace=True)
        tweets_df['tweet_text']=tweets_df['tweet_text'].apply(lambda x : p.clean(x))

        tweets_df['tweet_text'] = tweets_df['tweet_text'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
        tweets_df['tweet_text'] = tweets_df['tweet_text'].apply(lambda x:x.lower())
        tweets_df.drop_duplicates(inplace=True)
        tweets_df['tweet_text']=tweets_df['tweet_text'].apply(lambda x :re.sub(':','',x))
        nan_value = float("NaN")
        tweets_df.replace("", nan_value, inplace=True)
        tweets_df.dropna(inplace=True)
        tweets_df['subjectivity']=tweets_df['tweet_text'].apply(lambda x : TextBlob(x).sentiment.subjectivity)
        tweets_df['polarity']=tweets_df['tweet_text'].apply(lambda x : TextBlob(x).sentiment.polarity)
        tweets_df['label']=tweets_df['polarity'].apply(lambda x :'negative' if x<0 else ('positive' if x >0 else "neutral"))
        
        value=tweets_df.label.value_counts()
        i,j,k=value

        sentiment_new=sentiment(neutral=i,positive=j,negative=k)
        sentiment_new.save()
        print('data is saved in sentiment ')

        




        return i,j,k


        


