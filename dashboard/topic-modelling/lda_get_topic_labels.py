import tomotopy as tp
import pandas as pd

def get_tomotopy_topic_labels(topic_model, 
                              min_labeller_cf=10, min_labeller_df=5, 
                              min_labeller_len=1, max_labeller_len=5, 
                              max_labeller_cand=10000,
                              labeller_smoothing = 1e-2,
                              labeller_mu = 0.25,
                              labeller_max_n = 10 
                              ):
  """
  min_cf: minimum collection frequency of collocations. Collocations with a 
          smaller collection frequency than min_cf are excluded from the candidates. 
          Set this value large if the corpus is big 
          
  min_df: minimum document frequency of collocations. Collocations with a smaller
          document frequency than min_df are excluded from the candidates. 
          Set this value large if the corpus is big 
          
  max_len: maximum length of collocations 
  
  max_cand: int maximum number of candidates to extract
  
  smoothing: a small value greater than 0 for Laplace smoothing
  
  mu: a discriminative coefficient. Candidates with high score on a specific 
      topic and with low score on other topics get the higher final score when 
      this value is the larger.
  
  
  
  """
  import tomotopy as tp
  import pandas as pd

  
  
      # extract candidates for auto topic labeling
  extractor = tp.label.PMIExtractor(min_cf=min_labeller_cf, 
                                    min_df=min_labeller_df,
                                    min_len=min_labeller_len,
                                    max_len=max_labeller_len, 
                                    max_cand=max_labeller_cand, 
                                    normalized=True)
  cands = extractor.extract(topic_model)

  labeler = tp.label.FoRelevance(topic_model, cands, min_df=min_labeller_df, 
                                 smoothing=labeller_smoothing, mu=labeller_mu)
                                 
                                 
                                 
  label_df = pd.DataFrame(
    {
      f"Cluster {k}": 
          [label for label, _ in labeler.get_topic_labels(k, top_n=labeller_max_n)]
      for k in range(topic_model.k)
     })
     
  return(label_df)


if __name__ == '__main__':
  
  from lda_get_topic_model import get_topic_model
  
  from lda_pre_processing import *
  
  import pandas as pd
  
  reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())
  
  correct_reviews = preprocess_for_LDA(reviews)
  
  lda_model = get_topic_model(process_in(correct_reviews))
  
  labels = get_tomotopy_topic_labels(lda_model)
  
  print(labels)

