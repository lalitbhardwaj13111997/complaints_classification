import threading
from .models import tata_tweets,location
import pandas as pd
import tweepy
import re
import nltk
from nltk.corpus import stopwords
import preprocessor as p


class FetchApiThread(threading.Thread):

    def __init__(self,api_key,api_key_secret,access_token,access_token_secret):
        
        self.api_key=api_key
        self.api_key_secret=api_key_secret
        self.access_token=access_token
        self.access_token_secret=access_token_secret



        threading.Thread.__init__(self)
    
    


    def run(self):
        try:
            print('Thread execution started')
            auth=tweepy.OAuthHandler(self.api_key,self.api_key_secret)
            auth.set_access_token(self.access_token,self.access_token_secret)
            api=tweepy.API(auth)
            cursor=tweepy.Cursor(api.search_tweets,q='@tatamotors',tweet_mode='extended',lang='en').items()
            tatamotors_tweets=[]
            tweets_id=[]
            for i in cursor:
              loc=location(location=i.user.location) 
              loc.save()
              tatamotors_tweets.append(i.full_text) 

            df=pd.DataFrame(tatamotors_tweets,columns=['tweet_text'])
            df['tweet_text']=df['tweet_text'].apply(lambda x : p.clean(x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : re.sub(r"(?:\@|https?\://)\S+", "", x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : re.sub(r"(?:\#|https?\://)\S+", "", x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : re.sub('RT', "", x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : re.sub('\n', "", x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : re.sub('\d+', "", x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : re.sub('/', "", x))

            emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
            df['tweet_text']=df['tweet_text'].apply(lambda x : emoji_pattern.sub(r'',x))
            df['tweet_text']=df['tweet_text'].apply(lambda x : x.lower())
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True,inplace=True)
            

            
            # df.drop_duplicates(inplace=True)
            # stop=stopwords.words('english')
            # df['tweet_text']=df['tweet_text'].apply(lambda x : [item for item in x.split() if item not in stop]).apply(lambda x :" ".join(x))           


            for i in df['tweet_text']:
               tweet=tata_tweets(tweet_text=i) 
               tweet.save()

            
               
                




           
        except Exception as e:
            print(e)