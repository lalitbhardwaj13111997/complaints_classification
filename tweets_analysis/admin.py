from django.contrib import admin

# Register your models here.
from .models import models, tata_tweets,location,zconnect_tweets

admin.site.register(tata_tweets)
admin.site.register(location)
admin.site.register(zconnect_tweets)
