import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import re 

def get_graph():
    buffer=BytesIO()
    plt.savefig(buffer,format='png')
    buffer.seek(0)
    image=buffer.getvalue()
    graph=base64.b64encode(image)
    graph=graph.decode('utf-8')
    buffer.close
    return graph


def plot_graph(x,color):
    plt.switch_backend('AGG')
    plt.title('Sentiment Analysis')
    plt.xlabel('Sentiment')
    plt.ylabel('Count of Sentiments')
    plt.figure(figsize=(3,3))
    plt.hist(x,color=color)
    plt.tight_layout()
    graph=get_graph()
    return graph

def plot_graph_sentiment(df2):
    plt.switch_backend('AGG')
    plt.figure(figsize=(3,3))
    plt.bar(df2['Neutral'],height=220,width=15,label='Neutral',color='b')
    plt.bar(df2['Positive'],height=220,width=15,label='Positive',color='g')
    plt.bar(df2['Negative'],height=220,width=15,label='Negative',color='r')
    plt.xlabel("Tweets Sentiment Range ")
    plt.ylabel("Frequency of Tweets range")
    
    plt.legend()
    plt.tight_layout()
    graph=get_graph()
    return graph    

def plot_bar(x,y):
    plt.switch_backend('AGG')
   
    plt.figure(figsize=(5,5))
    plt.bar(x,y,color='maroon',width=0.4)
    plt.grid(True, color = "black", linewidth = "1", linestyle = "--")

    plt.title('Location Of Tweets')
    plt.xlabel('Cities (X-Axis)')
    plt.ylabel('No of Tweets (Y-Axis)')
    plt.tight_layout()
    graph=get_graph()
    return graph



def plot_pie(x,y,labels):
    slice=[x,y]
    plt.switch_backend('AGG')   
    plt.figure(figsize=(3,3))
    plt.pie(slice, labels = labels,autopct='%1.0f%%',pctdistance=0.5,labeldistance=1.2)
    plt.legend(title='Complaints Classification',bbox_to_anchor =(-0.01,0.1,0.7,1))
    graph=get_graph()
    return graph




def preprocessing(df):

            

    
           columns=['tweets']
           df.columns=columns

           df['tweet']=df['tweet'].apply(lambda x :re.sub(r"(?:\@|https?\://)\S+", "", x))
           df['tweet']=df['tweet'].apply(lambda x :re.sub(r"(?:\#|https?\://)\S+", "", x))
           df['tweet']=df['tweet'].apply(lambda x :re.sub('RT',"",x))
           df['tweet']=df['tweet'].apply(lambda x :re.sub('\n',"",x))
           df['tweet']=df['tweet'].apply(lambda x :re.sub("\d+", "",x))
           df['tweet']=df['tweet'].apply(lambda x :re.sub("/", "",x))


      
        
    
           return df

    
    



