from praw import Reddit

class RedditConnector(object):

    def __init__(self):
        super().__init__()
        self.reddit = Reddit(client_id = "",
					client_secret = "",
					user_agent = "",
					username = "",
					password = "")
    
    def get_url_details(self, url):
        return self.reddit.submission(url=url)

