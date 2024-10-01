
"""

question a dict of form {question_id: "Question Name"}

"""

import pickle, os, logging

import sys
sys.path.insert(0, './topic-modelling/')

from lda_pre_processing import preprocess_for_LDA, process_in

from lda_get_topic_model import get_topic_model

from lda_prepare_data import get_prepared_ldavis_data

from lda_get_topic_labels import get_tomotopy_topic_labels

from get_dominant_topic import get_dominant_topic

from re_generate_summaries import generate_topics_summary

from re_generate_sentiment import generate_topics_sentiment




# logger = logging.getLogger(__name__)
# 
# logging.basicConfig(filename='./logs/dashboard.log')

def generate_topic_model(processed_comments, n_topics, question_id=None, min_df=5, rm_top=5, training_iterations=1000):
  
  # 
  # base_path = "./cache/lda"
  # 
  # if question_id is None:
  #   path = base_path
  # else:
  #   path = f"{base_path}/{question_id}"
  # 
  # 
  # filename = f"topic_model_n={n_topics}_min-df={min_df}_rm-top={rm_top}_iters={training_iterations}"
  # 
  # filepath = f"{path}/{filename}"
  
  
    # Check if the file exists
  # if os.path.exists(filepath):
  #     # Load the contents from the file using pickle
  #     with open(filepath, 'rb') as file:
  #         model = pickle.load(file)
  #     logger.info(f"Topic Model loaded from existing file with params n={n_topics}, min_df={min_df}, rm_top={rm_top}")
  # else:
      # Run the function `get_topic_model()` to generate the topic model
  model = get_topic_model(processed_comments, n_topics, min_df, rm_top, training_iterations)
      # Ensure the directory exists
      # os.makedirs(path, exist_ok=True)
      # Save the data to the file using pickle
      # with open(filepath, 'wb') as file:
      #     pickle.dump(model, file)
      # logger.info(f"Topic Model generated and stored with params n={n_topics}, min_df={min_df}, rm_top={rm_top}")
      
  return model



def update_topic_model(comments, # v for lda
                       n_topics,  
                       min_df, rm_top, training_iterations,# v for labeller
                       min_labeller_cf, min_labeller_df,
                       min_labeller_len, max_labeller_len, 
                       max_labeller_cand,
                       labeller_smoothing,
                       labeller_mu,
                       labeller_max_n, # v for abs summary        
                       model_name,
                       pos_threshold, neg_threshold,
                       min_length, max_length, 
                       do_sample, temperature, 
                       length_penalty, num_beams, 
                       no_repeat_ngram_size,
                       num_extractive):
                         
  question_id=None
  R = 30 # Hardcode for now. Looks better
  
  processed_comments = preprocess_for_LDA(comments)
  
  lda_model = generate_topic_model(process_in(processed_comments), n_topics, question_id, min_df, rm_top, training_iterations)
  
  prepared_data = get_prepared_ldavis_data(lda_model, R)
  
  # tomotopy_labels = get_tomotopy_topic_labels(
  #                             lda_model, 
  #                             min_labeller_cf=10, min_labeller_df=5, 
  #                             min_labeller_len=1, max_labeller_len=5, 
  #                             max_labeller_cand=10000,
  #                             labeller_smoothing = 1e-2,
  #                             labeller_mu = 0.25,
  #                             labeller_max_n = 10 
  #                             )

  tomotopy_labels = get_tomotopy_topic_labels(
                              lda_model,
                              min_labeller_cf, min_labeller_df,
                              min_labeller_len, max_labeller_len,
                              max_labeller_cand,
                              labeller_smoothing,
                              labeller_mu,
                              labeller_max_n
                              )
                              
  
  dominant_topics_df = get_dominant_topic(processed_comments, lda_model)
                              
  abstractive_summaries, extractive_summaries = generate_topics_summary(dominant_topics_df, num_topics = lda_model.k)
  

  sentiment = generate_topics_sentiment(dominant_topics_df, num_topics = lda_model.k)
  
  
  return(prepared_data, tomotopy_labels, abstractive_summaries, extractive_summaries, sentiment)
