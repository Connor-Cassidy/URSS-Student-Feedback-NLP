import numpy as np
import pandas as pd


def get_dominant_topic(lda_preprocessed_comments, model):
  
  
    if (len(lda_preprocessed_comments) != len(model.docs)):
        raise AssertionError("Preprocessed input does not match in-built model pre-processing")
  
    
    
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in model.docs])
    doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
    
    topics_df = pd.DataFrame(doc_topic_dists, columns=[f"Cluster {i}" for i in range(model.k)])
  
    # Add the text labels as the first column
    topics_df.insert(0, 'Comment', lda_preprocessed_comments)
  
  
    cluster_columns = [col for col in topics_df.columns if col.startswith('Cluster')]
    
    ##| Identify dominant topic
    topics_df['Dominant_Topic'] = topics_df[cluster_columns].idxmax(axis=1)
    topics_df['Dominant_Topic'] = topics_df['Dominant_Topic'].apply(lambda x: int(x.split(' ')[-1]))
    
    return(topics_df[['Comment', 'Dominant_Topic']])
    
if __name__ == '__main__':
  
  from lda_get_topic_model import get_topic_model
  
  from lda_pre_processing import *
  
  reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())
  
  correct_reviews = preprocess_for_LDA(reviews)
  
  lda_model = get_topic_model(process_in(correct_reviews))
  
  dominant_topics = get_dominant_topic(correct_reviews, lda_model)
  
  print(dominant_topics)
