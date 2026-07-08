import re

def extract_hashtags_and_mentions(tweet):
    # \w+ matches alphanumeric characters and underscores
    hashtags = re.findall(r'#\w+', tweet)
    mentions = re.findall(r'@\w+', tweet)
    
    return {
        "hashtags": hashtags,
        "mentions": mentions
    }

if __name__ == '__main__':
    tweet = "Just deployed my new app! Thanks @OpenAI and @GitHub for the tools. #coding #python #webdev"
    print(extract_hashtags_and_mentions(tweet))
