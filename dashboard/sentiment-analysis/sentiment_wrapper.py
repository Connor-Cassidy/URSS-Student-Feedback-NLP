"""
This file contains a function acting as a wrapper for both positivity and subjectivity scores
"""

from positivity import extract_positivity
from subjectivity import extract_subjectivity

def get_sentiment_scores(comments):
  
  sentiment_df                 = extract_positivity(comments)
  
  sentiment_df["subjectivity"] = extract_subjectivity(comments)


  return sentiment_df



if (__name__ == "__main__"):
  
  import pandas as pd
  
  comments = (pd.read_csv("../education data/course_data_clean.csv")[['reviews']]).dropna() 

  print(get_sentiment_scores(comments))
