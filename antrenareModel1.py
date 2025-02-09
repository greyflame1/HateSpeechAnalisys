import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords 
import re   # stop words
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import seaborn as sns


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

data="train.csv"
df = pd.read_csv(data)
def statistici_descriptive(date):
    print("Statistici descriptive:")
    print(date.describe(include='all'))
statistici_descriptive(df)


text = list(df['tweet'])
labels = list(df['class'])

print(f"Total data inputs: {df.shape[0]}")

stop_words = set(stopwords.words('english'))


def remove_userHandles(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "", raw_text)
    return text
def remove_rt(raw_text):
    regex = r"RT"
    text = re.sub(regex, "", raw_text)
    return text

def remove_urls(raw_text):
    urls_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))"
    text = re.sub(urls_regex, "", raw_text)
    return text


def remove_punctuation(raw_text):
    # text = raw_text.replace(""", '')
    text = raw_text.replace("'", '')
    text = text.replace(".", '')
    text = text.replace(",", '')
    text = text.replace("!", '')
    text = text.replace(".", '')
    text = text.replace("?", '')
    text = text.replace("..", '')
    text = text.replace("...", '')
    
    return text

 
def remove_html(raw_text):
        html_regex = r"&[^\s;]+;"
        text = re.sub(html_regex, "", raw_text)
        return text


def remove_stopwords(raw_text):
    tokenized_text = nltk.word_tokenize(raw_text)
    text = [word for word in tokenized_text if word.lower() not in stop_words]
    text = " ".join(text)
    return text

def clean_data(df):
    clean = []
    clean =[remove_userHandles(text) for text in df]
    clean = [remove_urls(text) for text in clean]
    clean = [remove_punctuation(text) for text in clean]
    clean = [remove_html(text) for text in clean]
    clean = [remove_punctuation(text) for text in clean]
    clean = [remove_rt(text) for text in clean]
    clean = [remove_stopwords(text) for text in clean]
    return clean

cleaned_text = clean_data(text)

x_train, x_test, y_train, y_test = train_test_split(cleaned_text, labels, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


model = LogisticRegression()  # or RandomForestClassifier
model.fit(x_train_vectorized, y_train)


y_pred = model.predict(x_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))


plt.figure(figsize=(10, 6))
plt.bar(['Train', 'Test'], [len(x_train), len(x_test)])
plt.title('Number of Samples in Train and Test Sets')
plt.ylabel('Number of Samples')
plt.show() 

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


cm = confusion_matrix(y_test, y_pred)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Non-Offensive', 'Offensive', 'Hateful'])


cmd.plot(cmap=plt.cm.Reds)

plt.title('Confusion Matrix')
plt.show()

print("Confuzie matrice (fara valori):")
print("Non-Offensive | Offensive | Hateful")


def distributie_clase(date):
    print("Distributia claselor:")
    distributie = date['class'].value_counts()
    for label, count in distributie.items():
        print(f"{label}: {count}")

distributie_clase(df)


def matrice_corelatie(date):
    print("Matricea de corelatie:")
    date_numerice = date.select_dtypes(include=[np.number])
    corelatie = date_numerice.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corelatie, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Matricea de corelatie')
    plt.show()


matrice_corelatie(df)

joblib.dump(model, 'model_clasificare.joblib')
joblib.dump(vectorizer, 'vectorizator.joblib')

print(f"Model Accuracy: {accuracy * 100:.2f}%")
