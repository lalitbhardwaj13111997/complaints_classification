from unittest.util import _MAX_LENGTH
from django.db import models

# Create your models here.


class tata_tweets(models.Model):

    
    tweet_text=models.CharField(max_length=10000)
   


class location(models.Model):

    
    location=models.CharField(max_length=10000)



class zconnect_tweets(models.Model):
    
    zconnect_tweets=models.CharField(max_length=10000)


class sentiment(models.Model):
    neutral=models.IntegerField(default=0)
    positive=models.IntegerField(default=0)
    negative=models.IntegerField(default=0)

class model_accuracy_complaints(models.Model):
    model_accuracy=models.FloatField(default=0)    
    complaints=models.IntegerField(default=0)


      