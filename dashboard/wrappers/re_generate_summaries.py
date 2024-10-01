import sys
import pandas as pd

sys.path.insert(0, './summarization/')

from abstractive_summarization import summarize_by_sentiment
from extractive_summarization_tex_rank import TexRank


                           
def generate_total_summary(comments, 
                           model_name="facebook/bart-large-cnn",
                           pos_threshold=0.5, neg_threshold=-0.5,
                           min_length = 30, max_length= 150, 
                           do_sample=False, temperature=1, 
                           length_penalty=2, num_beams=8, 
                           no_repeat_ngram_size=3,
                           num_extractive=50):
                             
                             
  extractive_summary = TexRank(comments)['Comment'][:num_extractive]                           
  
  print('done extractivbe')
  
  abs_summary = summarize_by_sentiment(extractive_summary[0:50],
                         model_name, 
                         pos_threshold, neg_threshold,
                         min_length, max_length, 
                         do_sample, temperature, 
                         length_penalty, num_beams, 
                         no_repeat_ngram_size)
  
  return(abs_summary, extractive_summary)
    
    
                                  

    

def generate_topics_summary(dominant_topic_df, num_topics,
                                model_name="facebook/bart-large-cnn",
                                pos_threshold=0.5, neg_threshold=-0.5,
                                min_length = 30, max_length= 150, 
                                do_sample=False, temperature=1, 
                                length_penalty=2, num_beams=8, 
                                no_repeat_ngram_size=3,
                                num_extractive=50):
  abs_summaries = pd.DataFrame(None, index=range(num_topics), columns=["total", "pos", "neg"])                            
  
  max_len = max([len(dominant_topic_df[dominant_topic_df['Dominant_Topic'] == k]['Comment']) for k in range(num_topics)])
  
  ext_summaries = pd.DataFrame(index = range(max_len), columns = [f"Topic {i}" for i in range(num_topics)])  
  
  

                                  
  for k in range(num_topics):
    cur_topics_comments = dominant_topic_df[dominant_topic_df['Dominant_Topic'] == k]['Comment']
    
    
    if len(cur_topics_comments) != 0:
    
      cur_extractive_summary = TexRank(cur_topics_comments)['Comment'][:num_extractive]
    
    
      ext_summaries[f"Topic {k}"] =  pd.Series(cur_extractive_summary).reset_index(drop=True)
    
      abs_summaries.iloc[k] = summarize_by_sentiment(cur_extractive_summary[:50],
                                              model_name,
                                              pos_threshold, neg_threshold,
                                              min_length, max_length,
                                              do_sample, temperature,
                                              length_penalty, num_beams,
                                              no_repeat_ngram_size)
  return(abs_summaries, ext_summaries)
        
        
     


def update_abstractive_global():
  """
  Called whenever abstractive hyperparams (eg temp) are updated
  """
  pass
  
def update_abstractive_topics():
  """
  Called both when abstractive hyperparams are updated, and when LDA hyperparams are updated
  """
  pass
  
  
if __name__ == '__main__':
  
  sys.path.insert(0, './topic-modelling/')
  
  from lda_get_topic_model import get_topic_model
  
  from lda_pre_processing import *
  
  from get_dominant_topic import get_dominant_topic
  
  reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())
  
  correct_reviews = preprocess_for_LDA(reviews)
  
  lda_model = get_topic_model(process_in(correct_reviews))
  
  dominant_topics = get_dominant_topic(correct_reviews, lda_model)
  
  generate_abstractive_topics(dominant_topics[dominant_topics['Dominant_Topic']==0], 1)
