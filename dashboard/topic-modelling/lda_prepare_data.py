import numpy as np  
import pyLDAvis

def get_prepared_ldavis_data(lda_model, num_terms_to_display=30):
  """
  The number of terms to display indicates the number of bars present on the
  visualisation
  """
  # Use tomotopy recommended preprocessing.


  topic_term_dists = np.stack([lda_model.get_topic_word_dist(k) for k in range(lda_model.k)])
  doc_topic_dists = np.stack([doc.get_topic_dist() for doc in lda_model.docs])
  doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
  doc_lengths = np.array([len(doc.words) for doc in lda_model.docs])
  vocab = list(lda_model.used_vocabs)
  term_frequency = lda_model.used_vocab_freq

  prepared_data = pyLDAvis.prepare(
    topic_term_dists, 
    doc_topic_dists, 
    doc_lengths, 
    vocab, 
    term_frequency,
    start_index=0, # tomotopy starts topic ids with 0, pyLDAvis with 1
    sort_topics=False, # IMPORTANT: otherwise the topic_ids between pyLDAvis and tomotopy are not matching!
    n_jobs=1,
    R = num_terms_to_display # Number of terms to display. Reccomended to be between 10 and 50.
  )
  
  return prepared_data


if __name__ == '__main__':
  
  from lda_get_topic_model import get_topic_model
  
  from lda_pre_processing import *
  
  import pandas as pd
  
  reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())
  
  correct_reviews = preprocess_for_LDA(reviews)
  
  lda_model = get_topic_model(process_in(correct_reviews))
  
  
  prepared_ldavis = get_prepared_ldavis_data(lda_model, num_terms_to_display=30)
  pyLDAvis.save_html(prepared_ldavis, 'lda.html')
