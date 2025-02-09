import pandas as pd
import joblib
import nltk
import re
from nltk.corpus import stopwords
import unicodedata


nltk.download('punkt')
nltk.download('stopwords')


model = joblib.load('model_clasificare.joblib')
vectorizer = joblib.load('vectorizator.joblib')


stop_words = set(stopwords.words('english'))

def remove_userHandles(raw_text):
    regex = r"@([^ ]+)"
    return re.sub(regex, "", raw_text)

def remove_rt(raw_text):
    regex = r"RT"
    return re.sub(regex, "", raw_text)

def remove_urls(raw_text):
    urls_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))"
    return re.sub(urls_regex, "", raw_text)

def remove_punctuation(raw_text):
    text = raw_text.replace("'", '')
    text = text.replace(".", '')
    text = text.replace(",", '')
    text = text.replace("!", '')
    text = text.replace("?", '')
    return text

def remove_html(raw_text):
    html_regex = r"&[^\s;]+;"
    return re.sub(html_regex, "", raw_text)

def remove_stopwords(raw_text):
    tokenized_text = nltk.word_tokenize(raw_text)
    return " ".join([word for word in tokenized_text if word.lower() not in stop_words])

def remove_diacritics(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def preprocess_text(text):
    text = remove_userHandles(text)
    text = remove_urls(text)
    text = remove_punctuation(text)
    text = remove_html(text)
    text = remove_rt(text)
    text = remove_stopwords(text)
    return text


data_file = "date_noi_tweeter.csv"
df = pd.read_csv(data_file)


if 'tweet' not in df.columns:
    print("Coloana 'tweet' nu exista in fisierul CSV.")
else:

    df['text_procesat'] = df['tweet'].apply(preprocess_text)

    
    vectorized_text = vectorizer.transform(df['text_procesat'])

 
    predictions = model.predict(vectorized_text)

   
    labels = ['Non-Offensive', 'Offensive', 'Hateful']
    df['predicție'] = [labels[pred] for pred in predictions]

   
    print(df[['tweet', 'predicție']])


input_text = input("Introduceti tweet-ul pe care doriti sa-l clasificati: ")


if 'tweet' not in df.columns:
    print("Coloana 'tweet' nu exista in fisierul CSV.")
else:
   
    print(df[['tweet', 'predicție']])
