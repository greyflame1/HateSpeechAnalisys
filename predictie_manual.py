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

# Adăugăm o listă de cuvinte neutre
cuvinte_neutre = set(["cuvânt1", "cuvânt2", "cuvânt3"])  # Adaugă cuvintele neutre relevante

def verifica_cuvinte_neutre(text):
    # Verificăm dacă textul conține cuvinte neutre
    for cuvant in text.split():
        if cuvant.lower() in cuvinte_neutre:
            return True
    return False

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


input_text = input("Introduceti tweet-ul pe care doriti sa-l clasificati: ")

# Verificăm dacă textul conține cuvinte neutre
if verifica_cuvinte_neutre(input_text):
    print("Predicție: Neutru")  # Returnăm "Neutru" dacă textul conține cuvinte neutre
else:
    processed_text = preprocess_text(input_text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    labels = ['Non-Offensive', 'Offensive', 'Hateful']
    print(f"Predicție: {labels[prediction[0]]}")
