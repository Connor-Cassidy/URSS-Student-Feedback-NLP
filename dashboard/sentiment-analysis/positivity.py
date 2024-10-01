"""
This file contains a function which takes in a pandas dataframe of the form

 |   col_1  | 
 ------------
 | text_1 |
 | text_2 |
 |    :   |
 | text_N |
 
and maps each text_j to a sentiment score sentiment_j. To classify the
sentiment of each text, we use VADER, which can be found at 
https://github.com/nltk/nltk/blob/develop/nltk/sentiment/vader.py#L441 and is
published at https://doi.org/10.1609/icwsm.v8i1.14550. VADER is a rule based
model used specifically for social media text, which we claim to be broadly
similar to education reviews made by students.

This model does not require any pre-processing, as stop-word removal is in-built
and punctuation/capitalisation is used to capture sentiment.


This sentiment score is the sentiment object output by VADER, a dictionary of 
the form{'neg': neg_score, 'pos' : pos_score, 'neu': neu_score, 
'compound': compound_score}. Here, neg, pos and neu are scores ranging from 0 to
1, which can broadly be interpreted as the percentage of the text classified as
negative, positve and neutral respectively. The final compound score ranges from
-1 to 1, indicating the overall sentiment of the text, with values close to -1 
indicating the text was classified as very negative, values close to 0 indicating
neutrality and values close to 1 indicating positivity. The output dataframe is
thus of the form.

 |  neg    |  neu   |   pos   | compound  |
 ------------------------------------------
 |         |        |         |           |

 
Each text_j is expected to be one review, with each column either representing
a question or topic
 
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd


def extract_positivity(input_dataframe):
  
  # First, ensure the input is of correct form, and coerce to string if not.
  
  if (input_dataframe.map(lambda x: not isinstance(x, str)).any().any()):
    warn(f"Some cell in {input_dataframe} is not a string, attempting to coerce to string")
    input_dataframe = input_dataframe.map(str)
  
  # Initialise the sentiment analyzer and apply it to each cell, 
  # then return the new df
    
  analyzer = SentimentIntensityAnalyzer()
  
  # pd.json_normalise( - .iloc[:,0]) converts the list of dicts - to a df
  
  return(pd.json_normalize(input_dataframe.map(analyzer.polarity_scores).iloc[:,0]))
  

# from textblob import TextBlob
# 
# def extract_sentiment_textblob(input_dataframe):
#   
#   analysis = [TextBlob(i).sentiment.polarity > 0 for i in input_dataframe]
#   
#   return(analysis)
# 
# 
# from transformers import pipeline
# 
# text = comments['reviews']


# def extract_sentiment_huggingface(text):
#   classifier = pipeline('sentiment-analysis', device="cuda")
# 
#   result = pd.DataFrame([classifier(i)[0]['label'] for i in text])
#   #Returns TRUE for positive sentiment, FALSE for negative sentiment
#   
#   out = []
# 
#   for i in text:
#     out.append(pd.DataFrame(classifier(i))['label'])
    
  
  # return result == 'POSITIVE'
  
  # return classifier(text[1])[0]['label']

if (__name__ == "__main__"):
  
  import pandas as pd
  
  # Use 5 demo sentences taken from https://blog.quantinsti.com/vader-sentiment/ for first column
  
  sentences = ["Naman is smart, boring, and creative.",  # base example
             "Naman is smart, boring, and creative!!!!!!",  # using punctuations
             "Naman is SMART, boring, and CREATIVE.",  # using CAPS
             "Naman is smart, boring, and super creative.",  # using degree modifiers (example 1)
             "Naman is smart, slightly boring, and creative.",  # using degree modifiers (example 2)
             "Naman is smart, boring, and creative but procrastinates a lot.",  # Polarity shift due to Conjunctions
             "Naman isn't smart at all.",  # Catching Polarity Negation
             ]
             
  # other demo text taken as the first 7 test reviews from
  # https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?resource=download
  
  
  comments = (pd.read_csv("../education data/course_data_clean.csv")[['reviews', 'course_rating']]).dropna()        
  
  comments['positive'] = comments['course_rating'] == 'liked course'
  
  set(comments)
             
  combined_review_df = pd.DataFrame({
    # "social media posts": sentences,
    # "amazon reviews": reviews,
    'student': comments['reviews']
    })
    

    
  
  # diff_sent =pd.DataFrame(    {"true": list(comments['positive']),
  #   "vader": list(pd.json_normalize(extract_positivity(comments['reviews']))['compound'] >= 0),
    # "blob": extract_sentiment_textblob(comments['reviews']),
    # 'hugg': [extract_sentiment_huggingface(comments['reviews'][:7500]),
             # extract_sentiment_huggingface(comments['reviews'][7501:])]
  # })
                
  # from sklearn.metrics import f1_score, precision_score, recall_score
  #              
  # 
  # def compute(model):
  #   return(
  #   {
  #     'accuracy': sum(diff_sent['true'] == diff_sent[model])/ len(diff_sent),
  #     'f1':   f1_score(y_true = diff_sent['true'], y_pred = diff_sent[model]),
  #     'precision':   precision_score(y_true = diff_sent['true'], y_pred = diff_sent[model]),
  #     'recall':   recall_score(y_true = diff_sent['true'], y_pred = diff_sent[model])
  #   }
  #   )
    
  # print(f"vader: {compute('vader')}")
  # print(f"blob: {compute('blob')}")

