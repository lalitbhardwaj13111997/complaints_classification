
from . import views
from django.urls import path

urlpatterns = [
    


    path('',views.home,name='home'),
    path('tweets',views.tweets,name='tweets'),
    path('recent_tweets',views.recent_tweets,name='recent_tweets')

    

]
