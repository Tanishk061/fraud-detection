import nltk
nltk.download('stopwords')
import pickle
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

import re
import string
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_sms(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]
    
    return "Spam" if result == 1 else "Not Spam"
print(predict_sms("Win a free lottery now!!!"))