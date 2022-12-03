from logging import exception
import threading
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
from .models import model_accuracy_complaints


#####1 not complaint ### 0 complaint 

####1 is positive ###0  is sad 
class modelmtweets:




    # def document_vector(self,doc):
    # # remove out-of-vocabulary words
              
    #           model = gensim.models.Word2Vec(window=10,min_count=2)
 
    #           doc = [word for word in doc.split() if word in model.wv.index_to_key]
              
    #           return np.mean(model.wv[doc], axis=0)

    # def tweets_model(self):

    #     df=pd.read_csv('/home/lalit/Tata_cars_sentiment/tweets_analysis/zconnect_tweets.csv')
    #     df.drop(columns=['Unnamed: 0'],inplace=True)


    #     sw_list = stopwords.words('english')

    #     df['tweets'] = df['tweets'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
    #     df['tweets'] = df['tweets'].apply(lambda x:x.lower())

    
    #     story = []

    #     for doc in df['tweets']:

    #         raw_sent = sent_tokenize(doc)
    #         for sent in raw_sent:
    #             story.append(simple_preprocess(sent))
            

        
    #     model = gensim.models.Word2Vec(window=10,min_count=2)

    #     model.build_vocab(story)

    #     model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
    #     print(len(model.wv.index_to_key))

        


    #     X = []

    #     for doc in tqdm(df['tweets'].values):
    #          doc = [word for word in doc.split() if word in model.wv.index_to_key]
    #          value= np.mean(model.wv[doc], axis=0)
    #          X.append(value)

            

             



    #     X = np.array(X)
    #     print(X[0])


    #     encoder = LabelEncoder()

    #     y = encoder.fit_transform(df['labels'])

    #     print(y)
    #     X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=41)


    #     rf = RandomForestClassifier(n_estimators=102)
    #     rf.fit(X_train,y_train)

    #     query="""select * from tweets_analysis_tata_tweets"""
    #     conn=pg.connect(dbname="tweets",user="postgres",password="1234",host="localhost",port="5432")


    #     tweets_df=sql.read_sql_query(query,conn)
    #     tweets_df.drop(columns=['id'],inplace=True)
    #     tweets_df['tweet_text']=tweets_df['tweet_text'].apply(lambda x : p.clean(x))
    #     tweets_df['tweet_text']=tweets_df['tweet_text'].apply(lambda x : p.clean(x))

    #     tweets_df['tweet_text'] = tweets_df['tweet_text'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
    #     tweets_df['tweet_text'] = tweets_df['tweet_text'].apply(lambda x:x.lower())

    
    #     story2 = []

    #     for doc in tweets_df['tweet_text']:

    #         raw_sent = sent_tokenize(doc)
    #         for sent in raw_sent:
    #             story2.append(simple_preprocess(sent))
            

        
    #     model2 = gensim.models.Word2Vec(window=10,min_count=2)

    #     model2.build_vocab(story2)

    #     model2.train(story2, total_examples=model2.corpus_count, epochs=model2.epochs)
    #     print(len(model2.wv.index_to_key))

        


    #     X1 = []

    #     for doc in tqdm(tweets_df['tweet_text'].values):
    #          doc = [word for word in doc.split() if word in model2.wv.index_to_key]
    #          value= np.mean(model2.wv[doc], axis=0)
    #          X1.append(value)

            

             



    #     X1 = np.array(X1)
    #     print(X1[0])

    #     y_pred = rf.predict(X1)

    #     positive=0
    #     sad=0


    #     for i in y_pred:
    #         if i == 1:
    #             positive=positive+1;

    #         elif i == 0:
    #             sad=sad+1;

    #     return positive,sad        





    def domodel(self):


        df=pd.read_csv('/home/lalit/Tata_cars_sentiment/tweets_analysis/zconnect_tweets.csv')
        df.drop(columns=['Unnamed: 0'],inplace=True)


        sw_list = stopwords.words('english')

        df['tweets'] = df['tweets'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
        df['tweets'] = df['tweets'].apply(lambda x:x.lower())

    
        story = []

        for doc in df['tweets']:

            raw_sent = sent_tokenize(doc)
            for sent in raw_sent:
                story.append(simple_preprocess(sent))
            

        
        model = gensim.models.Word2Vec(window=10,min_count=2)

        model.build_vocab(story)

        model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
        print(len(model.wv.index_to_key))

        


        X = []

        for doc in tqdm(df['tweets'].values):
             doc = [word for word in doc.split() if word in model.wv.index_to_key]
             value= np.mean(model.wv[doc], axis=0)
             X.append(value)

            

             



        X = np.array(X)
        print(X[0])


        encoder = LabelEncoder()

        y = encoder.fit_transform(df['labels'])

        print(y)
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=41)


        rf = RandomForestClassifier(n_estimators=102)
        rf.fit(X_train,y_train)
        y_pred = rf.predict(X_test)
        print(X_test)
        score=accuracy_score(y_test,y_pred)*100
        answer = str(round(score, 2))




        conn=pg.connect(dbname="tweets",user="postgres",password="1234",host="localhost",port="5432")

        

                ###for the normal tweets 


        query="""select * from tweets_analysis_zconnect_tweets"""

        tweets_df=sql.read_sql_query(query,conn)
        tweets_df.drop(columns=['id'],inplace=True)
        tweets_df['zconnect_tweets']=tweets_df['zconnect_tweets'].apply(lambda x : p.clean(x))
        tweets_df['zconnect_tweets']=tweets_df['zconnect_tweets'].apply(lambda x : p.clean(x))

        tweets_df['zconnect_tweets'] = tweets_df['zconnect_tweets'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
        tweets_df['zconnect_tweets'] = tweets_df['zconnect_tweets'].apply(lambda x:x.lower())

    
        story1 = []

        for doc in tweets_df['zconnect_tweets']:

            raw_sent = sent_tokenize(doc)
            for sent in raw_sent:
                story1.append(simple_preprocess(sent))
            

        
        model1 = gensim.models.Word2Vec(window=10,min_count=2)

        model1.build_vocab(story)

        model1.train(story1, total_examples=model1.corpus_count, epochs=model1.epochs)
        print(len(model1.wv.index_to_key))

        


        X1 = []

        for doc in tqdm(tweets_df['zconnect_tweets'].values):
             doc = [word for word in doc.split() if word in model1.wv.index_to_key]
             value= np.mean(model1.wv[doc], axis=0)
             X1.append(value)

            

             



        X1 = np.array(X1)
        print(X1[0])

        y_pred = rf.predict(X1)

        values=0

        for i in y_pred:
            if i == 0:
                values=values+1

        model_accuracy=model_accuracy_complaints(model_accuracy=answer,complaints=values)  
        model_accuracy.save()
        print('Model accuracy db saved')    





        # query1="""select * from tweets_analysis_tata_tweets"""

        # tweets_nexon=sql.read_sql_query(query1,conn)
        # tweets_nexon.drop(columns=['id'],inplace=True)
        # tweets_nexon['tweet_text']=tweets_nexon['tweet_text'].apply(lambda x : p.clean(x))
        # tweets_nexon['tweet_text']=tweets_nexon['tweet_text'].apply(lambda x : p.clean(x))

        # tweets_nexon['tweet_text'] = tweets_nexon['tweet_text'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
        # tweets_nexon['tweet_text'] = tweets_nexon['tweet_text'].apply(lambda x:x.lower())

    
        # story3 = []

        # for doc in tweets_nexon['tweet_text']:

        #     raw_sent = sent_tokenize(doc)
        #     for sent in raw_sent:
        #         story3.append(simple_preprocess(sent))
            

        
        # model3 = gensim.models.Word2Vec(window=10,min_count=2)

        # model3.build_vocab(story3)

        # model3.train(story3, total_examples=model3.corpus_count, epochs=model3.epochs)
        # print(len(model3.wv.index_to_key))

        


        # X3 = []

        # for doc in tqdm(tweets_nexon['tweet_text'].values):
        #      doc = [word for word in doc.split() if word in model3.wv.index_to_key]
        #      value= np.mean(model3.wv[doc], axis=0)
        #      X3.append(value)

            

             



        # X3 = np.array(X3)
        # print(X3[0])

        # y_pred = rf.predict(X3)

        # sad=0

        # for i in y_pred:
        #     if i == 0:
        #         sad=sad+1   







        return answer,values;




            

###steps
#1 preprocess
#2 vectorization,tokenization
#3 model training
#4 accuracy
#5 take data from db and apply on the model



