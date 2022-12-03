from .thread import FetchApiThread    
    
    

def fetch():
    #print('fetch tweets started ')

    api_key='cW4RFZ9W4bDQ3Bd8EWvxsxQYd'
    api_key_secret='VIM3vnlY8dLNUMSnAYQfkew26OElxCpLTKRhmqsxP27IFVK7Ly'
    access_token='1564498476242800640-Zc2AQeGkwv4BIlS2IGFvOFns0nOXwK'
    access_token_secret='Cy37bmWjsQfQ84JmmvyBVeaqNxx2gmAv3IuChwslLLgA2'
    FetchApiThread(api_key,api_key_secret,access_token,access_token_secret).start()

 
   



  