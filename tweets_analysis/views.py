
from http import client
import imp
from pydoc import cli
from django.shortcuts import render
import tweepy
import pandas as pd
from .models import models,tata_tweets,location,zconnect_tweets
import re
import numpy as np
from .utils import plot_graph, plot_graph_sentiment,plot_pie,plot_bar
from .thread import FetchApiThread
import config
import GetOldTweets3 as got
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from .utils import preprocessing
import psycopg2 as pg
import pandas.io.sql as sql
import warnings
from googletrans import Translator
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import gensim
from .machinelearning import modelmtweets
from .sentiment import sentiment1
from sklearn.metrics import accuracy_score
import smtplib
from reportlab.pdfgen import canvas  
from django.http import HttpResponse 


    
# Create your views here.
def tweets(request):
    warnings.filterwarnings('ignore')

    # api_key='cW4RFZ9W4bDQ3Bd8EWvxsxQYd'
    # api_key_secret='VIM3vnlY8dLNUMSnAYQfkew26OElxCpLTKRhmqsxP27IFVK7Ly'
    # access_token='1564498476242800640-Zc2AQeGkwv4BIlS2IGFvOFns0nOXwK'
    # access_token_secret='Cy37bmWjsQfQ84JmmvyBVeaqNxx2gmAv3IuChwslLLgA2'
    # auth=tweepy.OAuthHandler(api_key,api_key_secret)
    # auth.set_access_token(access_token,access_token_secret)
    # api=tweepy.API(auth)
    # cursor=tweepy.Cursor(api.search_tweets,q='#zconnect',tweet_mode='extended').items()
    # tatamotors=[]
    # for i in cursor:
    #     tatamotors.append(i.full_text)

    #     df=pd.DataFrame(tatamotors)

    #     df=preprocessing(df)
    

    conn=pg.connect(dbname="tweets",user="postgres",password="1234",host="localhost",port="5432")
    query="""select * from tweets_analysis_location"""
    df=sql.read_sql_query(query,conn)
    df.drop(columns=['id'],inplace=True)
    df.dropna(inplace=True)
    counts=df['location'].value_counts()
    df.dropna(inplace=True)
    df['location']=df['location'].apply(lambda x :re.sub('India','',x))
    df['location']=df['location'].apply(lambda x :re.sub(',','',x))
    df['location']=df['location'].apply(lambda x :re.sub(' ','',x))
    df.replace('',np.nan,inplace=True)
    df.dropna(inplace=True)
    df_new=df['location'].value_counts().head(5)
    df1=pd.DataFrame(df_new)
    tl=Translator()
    list1=list(df1.index)
    df2=pd.DataFrame(list1,columns=['location'])
    df2['location']=df2['location'].apply(lambda x : tl.translate(x).text)
    count=df1['location'].values

    bar=plot_bar(x=df2['location'],y=count)



    

    return render(request,'tweets.html',{"bar":bar})




def home(request):

    warnings.filterwarnings('ignore')
    tweets_model=modelmtweets()

   


    api_key='cW4RFZ9W4bDQ3Bd8EWvxsxQYd'
    api_key_secret='VIM3vnlY8dLNUMSnAYQfkew26OElxCpLTKRhmqsxP27IFVK7Ly'
    access_token='1564498476242800640-Zc2AQeGkwv4BIlS2IGFvOFns0nOXwK'
    access_token_secret='Cy37bmWjsQfQ84JmmvyBVeaqNxx2gmAv3IuChwslLLgA2'
    #FetchApiThread(api_key,api_key_secret,access_token,access_token_secret).start()
    conn=pg.connect(dbname="tweets",user="postgres",password="1234",host="localhost",port="5432")
    query="""select * from tweets_analysis_location"""
    df=sql.read_sql_query(query,conn)
    df.drop(columns=['id'],inplace=True)
    df.dropna(inplace=True)
    counts=df['location'].value_counts()
    df.dropna(inplace=True)
    df['location']=df['location'].apply(lambda x :re.sub('India','',x))
    df['location']=df['location'].apply(lambda x :re.sub(',','',x))
    df['location']=df['location'].apply(lambda x :re.sub(' ','',x))
    df.replace('',np.nan,inplace=True)
    df.dropna(inplace=True)
    df_new=df['location'].value_counts().head(5)
    df1=pd.DataFrame(df_new)
    tl=Translator()
    list1=list(df1.index)
    df2=pd.DataFrame(list1,columns=['location'])
    df2['location']=df2['location'].apply(lambda x : tl.translate(x).text)
    location=df2['location'][0]
    lst = ['positive', 'negative', 'positive', 'negative',
            'positive', 'negative', 'positive']


    lst2 = ['delhi', 'delhi', 'mumbai', 'mumbai',
            'mumbai', 'goa', 'goa','thane','thane','gujrat','delhi','goa']       

    df = pd.DataFrame(lst,columns=['sentiment'])

    df1=pd.DataFrame(lst2,columns=['location'])

    mylabels = ["complaints","non-complaints"]

    chart1_color='red'
    chart2_color='purple'
    
    chart1=plot_graph(df1['location'],chart2_color)



    accuracy_query="""select * from tweets_analysis_model_accuracy_complaints"""
    accuracy_df=sql.read_sql_query(accuracy_query,conn)
    score=accuracy_df['model_accuracy'][0]
    value=accuracy_df['complaints'][0]
  
    

    sentiment_query="""select * from tweets_analysis_sentiment"""
    sentiment_df=sql.read_sql_query(sentiment_query,conn)
    neutral=sentiment_df['neutral'][0]
    positive=sentiment_df['positive'][0]
    negative=sentiment_df['negative'][0]





  
    df2 = pd.DataFrame({"Neutral":[neutral],"Positive":[positive],"Negative":[negative]})
    chart=plot_graph_sentiment(df2=df2)

    pie=plot_pie(x=negative,y=positive,labels=mylabels)

    server=smtplib.SMTP('smtp.gmail.com','587')
    server.starttls()
    server.login('lalit.bhardwaj13111997@gmail.com','edjzgoaptpbmtlwf')
    message = 'Subject: {}\n\n{}'.format("Zconnect Complaints", f'Zconnect complainst for today are {value}')
    server.sendmail('lalit.bhardwaj13111997@gmail.com','lsharma_me20@thapar.edu',message)
    print('mail sent ')
   
   

    # positive,sad=tweets_model.tweets_model()

    



   
 
# Calling DataFrame constructor on list
  
    # for i in df['twitter_id']:
      

    #     tweet_id=tata_tweets(twitter_id=i)   
    #     tweet_id.save() 


        
  
      

    return render(request,'home.html',{'chart':chart,'pie':pie,'chart1':chart1,'location':location,'neutral':neutral,'negative':negative,'positive':positive,'score':score,'value':value})    






      
def recent_tweets(request):
    warnings.filterwarnings('ignore')



    conn=pg.connect(dbname="tweets",user="postgres",password="1234",host="localhost",port="5432")

    query="""select * from tweets_analysis_zconnect_tweets"""
    tweets_df=sql.read_sql_query(query,conn)

    tweets_df.drop(columns=['id'],inplace=True)
    tweets_df.drop_duplicates(inplace=True)
    tweets_df.reset_index(drop=True,inplace=True)
    tweets_df
    emoji_pattern = re.compile("["
                          u"\U0001F600-\U0001F64F"  
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    tweets_df['zconnect_tweets']=tweets_df['zconnect_tweets'].apply(lambda x : emoji_pattern.sub(r'',x))
    tweets_df['tweet_text']=tweets_df['zconnect_tweets'].apply(lambda x : emoji_pattern.sub(',',x))
    tweets_df['zconnect_tweets']=tweets_df['zconnect_tweets'].apply(lambda x : emoji_pattern.sub(r' .',x))

    tweets_df['zconnect_tweets'] = tweets_df['zconnect_tweets'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')

    text=tweets_df.tail(8).values.item(1)
    text1=tweets_df.tail(8).values.item(2)
    text2=tweets_df.tail(8).values.item(3)
    text3=tweets_df.tail(8).values.item(4)
    text4=tweets_df.tail(8).values.item(5)
    text5=tweets_df.tail(8).values.item(6)
    text6=tweets_df.tail(8).values.item(7)
   




    return render(request,'recent_tweets.html',{'text':text,'text1':text1,'text2':text2,'text3':text3,'text4':text4,'text5':text5,'text6':text6})
 


