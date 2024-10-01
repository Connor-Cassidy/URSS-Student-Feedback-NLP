"""
NA RN, no hyperparams
"""

import pandas as pd
import sys
sys.path.insert(0, './sentiment-analysis/')

from sentiment_wrapper import get_sentiment_scores

def generate_topics_sentiment(dominant_topic_df, num_topics):
  ##| For each topic, return the mean sentiment of all comments which have that as their dominant topic.
  
  
  
  sentiment_by_topics = pd.DataFrame(None, index=range(num_topics), columns=["neg","neu","pos","compound", "subjectivity"])
  
  for k in range(num_topics):
    
    cur_topics_comments = dominant_topic_df[dominant_topic_df['Dominant_Topic'] == k]['Comment']
    
    sentiment_by_topics.iloc[k] = get_sentiment_scores(pd.DataFrame(cur_topics_comments)).mean()
    
  
  return(sentiment_by_topics)
  
  
def generate_total_sentiment(comments):
  
  return(get_sentiment_scores(pd.DataFrame(comments)).mean())
  


if __name__ == '__main__':
  
  comments = pd.DataFrame(['a','v'])
  
  print(generate_total_sentiment(comments))
