import nltk
from string import punctuation
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import pickle
import re
import RedditConnector as reddit_connector
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

class TextClassifier(object):

    def __init__(self):
        super().__init__()
        MODEL = 'logreg.pkl'
        COUNT = 'count_scaler.pkl'
        TFIDF = 'tdidf_scaler.pkl'
        TOKENIZER = 'tokenizer.pickle'
        LSTM = 'lstm_word2vec_model.h5'
    
        with open(MODEL, "rb") as file:
            self.logreg = pickle.load(file)

        with open(COUNT, "rb") as file:
            self.count = pickle.load(file)

        with open(TFIDF, "rb") as file:
            self.tfidf = pickle.load(file)

        with open(TOKENIZER, "rb") as file:
            self.tokenizer = pickle.load(file)

        self.lstm_model = load_model(LSTM)
        self.reddit = reddit_connector.RedditConnector()
        self.id_to_class = {
            0: "AskIndia",
            1: "Coronavirus",
            2: "Non-Political",
            3: "Policy/Economy",
            4: "Politics",
            5: "Science/Technology"
        }

    def clean_text(self, text):
        stop = stopwords.words('english') + ['http', 'https'] + ['nan', '[deleted]', '[removed]']
        text = re.sub('http[s]?://\S+', '', text)
        text = text.translate(str.maketrans('', '', punctuation))
        text = text.lower().strip()
        text = ' '.join([i if i not in stop and i.isalpha() else '' for i in text.lower().split()])
        text = re.sub(r"\s{2,}", " ", text)
        return text

    def logreg_predict_class(self, url):
        submission = self.reddit.get_url_details(url)
        data = {}
        data['title'] = str(submission.title)
        data['selftext'] = str(submission.selftext)
        data['link_flair_text'] = str(submission.link_flair_text)

        title = self.clean_text(data['title'])
        selftext = self.clean_text(data['selftext'])

        title = title + ' ' + selftext
        count_text = self.count.transform([title])
        tfidf_text = self.tfidf.transform(count_text)
    
        data['predicted_flair'] = self.logreg.predict(tfidf_text)[0]
        return data

    def lstm_predict_class(self, url):
        submission = self.reddit.get_url_details(url)
        data = {}
        data['title'] = str(submission.title)
        data['selftext'] = str(submission.selftext)
        data['link_flair_text'] = str(submission.link_flair_text)

        title = self.clean_text(data['title'])
        selftext = self.clean_text(data['selftext'])

        title = title + ' ' + selftext

        sequences = self.tokenizer.texts_to_sequences([title])
        padded_sequence = pad_sequences(sequences, maxlen=250)

        value = self.lstm_model.predict_classes(padded_sequence)[0]
        data['predicted_flair'] = self.id_to_class[value]
        return data

        