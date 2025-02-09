import tweepy
import pandas as pd
import csv

def citeste_credentiale():
    """Citeste credentialele din fisierul CSV"""
    try:
        with open('credentials.csv', 'r') as file:
            reader = csv.DictReader(file)
            # Afișăm numele coloanelor pentru debug
            print("Coloanele disponibile:", reader.fieldnames)
            for row in reader:
                return row
    except FileNotFoundError:
        print("Eroare: Fisierul credentials.csv nu a fost gasit!")
        return None
    except Exception as e:
        print(f"Eroare la citirea credentialelor: {str(e)}")
        return None

def configureaza_api():
    """Configureaza si returneaza API-ul Twitter"""
    # Citim credentialele
    credentials = citeste_credentiale()
    
    if not credentials:
        print("Nu s-au putut obtine credentialele!")
        return None
    
    try:
        # Autentificare cu Twitter
        auth = tweepy.OAuthHandler(
            credentials['api_key'],           # Modificat pentru a se potrivi cu CSV
            credentials['api_secret']         # Modificat pentru a se potrivi cu CSV
        )
        auth.set_access_token(
            credentials['access_token'],      # Modificat pentru a se potrivi cu CSV
            credentials['access_token_secret']# Modificat pentru a se potrivi cu CSV
        )
        
        # Cream obiectul API
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api
    except KeyError as e:
        print(f"Eroare: Lipseste cheia {e} din fisierul de credentiale!")
        return None
    except Exception as e:
        print(f"Eroare la configurarea API: {str(e)}")
        return None

def extrage_tweets(api, numar_tweets=100):
    """Extrage tweet-uri din Twitter"""
    if not api:
        print("API-ul nu a fost configurat corect!")
        return []
        
    tweets = []
    try:
        # Cautam tweet-uri in engleza
        for tweet in tweepy.Cursor(api.search_tweets,
                                 q="lang:en",
                                 tweet_mode="extended",
                                 lang="en").items(numar_tweets):
            
            # Extragem textul tweet-ului
            if hasattr(tweet, "retweeted_status"):
                text = tweet.retweeted_status.full_text
            else:
                text = tweet.full_text
                
            tweets.append({
                'tweet': text
            })
            
    except tweepy.TweepyException as e:
        print(f"Eroare la extragerea tweet-urilor: {str(e)}")
    
    return tweets

def main():
    # Configuram API-ul
    print("Configurare API Twitter...")
    api = configureaza_api()
    
    if not api:
        print("Nu se poate continua fara configurarea corecta a API-ului")
        return
    
    # Extragem tweet-urile
    print("Extragere tweet-uri...")
    tweets = extrage_tweets(api)
    
    # Salvam tweet-urile intr-un CSV
    if tweets:
        df = pd.DataFrame(tweets)
        df.to_csv('date_noi_tweeter.csv', index=False)
        print(f"Au fost extrase {len(tweets)} tweet-uri si salvate in date_noi_tweeter.csv")
    else:
        print("Nu au fost extrase tweet-uri")

if __name__ == "__main__":
    main()
