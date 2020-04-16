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

class TextClassifier(object):

    def __init__(self):
        super().__init__()
        MODEL = 'logreg.pkl'
        COUNT = 'count_scaler.pkl'
        TFIDF = 'tdidf_scaler.pkl'
    
        with open(MODEL, "rb") as file:
            self.logreg = pickle.load(file)

        with open(COUNT, "rb") as file:
            self.count = pickle.load(file)

        with open(TFIDF, "rb") as file:
            self.tfidf = pickle.load(file)

        self.reddit = reddit_connector.RedditConnector()

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

        