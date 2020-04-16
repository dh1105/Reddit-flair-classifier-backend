from praw import Reddit

class RedditConnector(object):

    def __init__(self):
        super().__init__()
        self.reddit = Reddit(client_id = "-xZ8AcnSY-RDGA",
					client_secret = "jUAVs7W9Qnh9ECET2VuG2bpjFx4",
					user_agent = "First app",
					username = "sgtcockmunch",
					password = "Snowy#150319")
    
    def get_url_details(self, url):
        return self.reddit.submission(url=url)

