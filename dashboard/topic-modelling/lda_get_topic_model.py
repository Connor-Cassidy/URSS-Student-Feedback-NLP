"""
This file contains a function which takes in a list of the form

[  1_text,  2_text, ..., N_text ]
 
 alongside the number of clusters K, and attempts to classify each review into 
 K many topics through latent dirilect allocation (LDA). The output from this 
 function is a pandas dataframe with N rows and K columns, where each cell contains
 the probability that review n belongs to cluster k.
 
"""
 


from nltk.corpus import stopwords
from nltk import PorterStemmer
 
import tomotopy as tp
import re

 




  
 
def get_topic_model(preprocessed_comments, num_topics=10, min_df=5, rm_top=5, training_iterations=1000): 
  
  # Use tomotopy recommended preprocessing.

  porter_stemmer = PorterStemmer().stem
  english_stops = set(porter_stemmer(w) for w in stopwords.words('english'))
  pat = re.compile('^[a-z]{1,}$')
  corpus = tp.utils.Corpus(
        tokenizer=tp.utils.SimpleTokenizer(porter_stemmer), 
        stopwords=lambda x: x in english_stops or not pat.match(x)
    )
    
  
  corpus.process(d.lower() for d in preprocessed_comments)


  model = tp.LDAModel(
    min_df = min_df, # Min document frequency, omit words which appear less than this many times
    rm_top = rm_top, # Remove the top this many words
    k = num_topics,
    corpus = corpus
  )
  

  
  if (len(preprocessed_comments) != len(model.docs)):
    ##| A Possible cause could be arabic text.
    ##| For some reason, python regex categorized arabic characters as latin characters
    raise AssertionError("Something failed in pre-processing. Good Luck.")



  model.train(training_iterations, show_progress=True)

  return model
 
 

  

      


# """
# selected_topic: int between 0 and n_topics - 1
# model: tp model obj
# """
# def get_topic_extractive_summary(sanitized_input, selected_topic, model, top_n=5):
#   
#   doc_topic_dists = np.stack([doc.get_topic_dist() for doc in model.docs])
#   doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
#   
#   
#   cur_topic_comments = pd.DataFrame({"Dominant Topic": np.argmax(doc_topic_dists, axis=1),
#                "Comment": sanitized_input})
#                
#   cur_topic_comments = cur_topic_comments[cur_topic_comments['Dominant Topic'] == selected_topic]['Comment']
#   
#   
#   
#   return()
# 
# 
# 
# # Calculate the cosine similarity between comment embeddings
# 
# 
#                       
# 
# 
# 
# 
# 
# """
# selected_topic: int between 0 and n_topics - 1
# model: tp model obj
# labelker: tp labeller obj
# **kwargs: to be passed to summarizer
# """
# def get_topic_summary(selected_topic, model, labeler, **kwargs):
#   
#   if (selected_topic not in range(model.k)):
#     raise ValueError(f"Topic num {selected_topic} is out of range for LDA with {model.k} topics")
#   
#   
#   labels = [label for label, score in labeler.get_topic_labels(selected_topic, **kwargs)]
#   
#   
#   sentiment_df = pd.json_normalize(extract_sentiment(pd.DataFrame(df_topic_dists["Text"])).iloc[:,0])
#   
#   comb = pd.concat([df_topic_dists, sentiment_df], axis=1)
#   
#   
# 
#   
#   
# 
#   get_topic_summary(selected_topic, model, labeler, top_n=6)
#                            
#   total_pos = comb['pos'].mean()
#   total_neg = comb['neg'].mean()
#   total_neu = comb['neu'].mean()
#  
#   total_pos + total_neg + total_neu
#  
#   for i in range(2):
#     print(i)
#  
#  
#  
#   topic_sentiments = pd.DataFrame({
#     "Cluster": [i for i in range(model.k)] + ['Total'],
#     "pos": np.zeros(model.k+1),
#     "neu": np.zeros(model.k+1),
#     "neg": np.zeros(model.k+1),
#     "compound": np.zeros(model.k+1),
#   })
#  
#   for i in range(model.k):
#     cur_col = f"Cluster {i}"
#     
#     positivity = comb[cur_col].dot(comb['pos'])
#     negativity = comb[cur_col].dot(comb['neg'])
#     neutrality = comb[cur_col].dot(comb['neu'])
#     
#     
#     new = comb[cur_col] / comb[cur_col].sum()
#     
#     compound = new.dot(comb['compound'])
#     
#     total = positivity + negativity + neutrality
#     
#     positivity /= total
#     negativity /= total
#     neutrality /= total
#     
#     print(f"Scores for Cluster {i} are| pos:{positivity}, neg:{negativity}, neu:{neutrality}")
#     
#     print(f"COMPOUND OF {compound}")
#     
#     topic_sentiments.iloc[i, 1:] = {"pos": positivity, "neu":neutrality, "neg":negativity, "compound":compound}
#     
#     # NOW DO IT OVERALL
# 
#   topic_sentiments.iloc[model.k] = {
#     "Cluster" : "Total", 
#     "pos"     : comb['pos'].mean(), 
#     "neu"     : comb['neg'].mean(), 
#     "neg"     : comb['neu'].mean(), 
#     "compound": comb['compound'].mean()
#     }
#     
#     
#     
#     cluster_columns = [col for col in comb.columns if col.startswith('Cluster')]
#     
#     comb['Selected_Topic'] = comb[cluster_columns].idxmax(axis=1)
#     comb['Selected_Topic'] = comb['Selected_Topic'].apply(lambda x: int(x.split(' ')[-1]))
# 
# 
#     new_df = comb[['Text', 'Selected_Topic', 'pos', 'neg', 'neu', 'compound']]



                                       
if (__name__ == "__main__"):

  import pandas as pd
             
  reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())
    
  print(get_topic_models(reviews))
