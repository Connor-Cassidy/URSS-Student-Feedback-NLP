###|
###| Load libraries and processing functions
###|

import pandas as pd
import sys
sys.path.insert(0, 'wrappers/')

from re_generate_topics import update_topic_model
from re_generate_sentiment import generate_total_sentiment, generate_topics_sentiment
from re_generate_summaries import generate_total_summary, generate_topics_summary

##|
##| Given hyperparameters below, process input data for use in dashboard
##|

model_name="facebook/bart-large-cnn"
pos_threshold=0.2
neg_threshold=-0.2
min_length = 30
max_length= 150
do_sample=False
length_penalty=2
num_beams=8
no_repeat_ngram_size=3
max_extractive=50
temperature=1


min_df=5
rm_top=5
training_iterations=10_000
n_topics=[6,8,10,12]

# v for labeller
min_labeller_cf=10
min_labeller_df=5
min_labeller_len=1
max_labeller_len=5
max_labeller_cand=10_000
labeller_smoothing=1e-2
labeller_mu=0.25
labeller_max_n=10

##|
##| Load in data
##|

# 500 means texrank wont converge?
comments = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())[0:600]

##|
##| Process LDA, abs / ext summary and sentiment analysis
##|

for n_topic in n_topics:
  prepared_data, tomotopy_labels, abstractive_summaries, extractive_summaries, sentiment =update_topic_model(comments, n_topic,   min_df, rm_top, training_iterations, min_labeller_cf, min_labeller_df,   min_labeller_len, max_labeller_len,   max_labeller_cand,  labeller_smoothing,  labeller_mu,  labeller_max_n,  model_name,  pos_threshold, neg_threshold,   min_length, max_length,   do_sample, temperature,  length_penalty, num_beams,   no_repeat_ngram_size,  max_extractive)
  
  prepared_data[0].to_csv(f'cache/topic_{n_topic}/coords.csv', index=False)
  tomotopy_labels.to_csv(f'cache/topic_{n_topic}/labels.csv', index=False)
  abstractive_summaries.to_csv(f'cache/topic_{n_topic}/abs.csv', index=False)
  extractive_summaries.to_csv(f'cache/topic_{n_topic}/ext.csv', index=False)
  sentiment.to_csv(f'cache/topic_{n_topic}/sent.csv', index=False)
              
                       

sent = generate_total_sentiment(comments)

abs, ext = generate_total_summary(comments, 
                       model_name,
                       pos_threshold, neg_threshold,
                       min_length, max_length, 
                       do_sample, temperature, 
                       length_penalty, num_beams, 
                       no_repeat_ngram_size,
                       num_extractive=50)
                       
pd.DataFrame(abs   , index=range(1)).to_csv(f'cache/total/abs.csv', index=False)
ext.to_csv(f'cache/total/ext.csv', index=False)
sent.to_csv(f'cache/total/sent.csv', index=False)
