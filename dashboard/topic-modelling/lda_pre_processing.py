import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

stop_words = set(stopwords.words('english'))

def check_stopwords(texts):
  
    pattern = re.compile(r'[^A-Za-z\s]*')

    
    no_white = [i.strip() for i in texts]
    
    no_stopwords = [' '.join([word for word in simple_preprocess(str(doc)) if word not in stop_words]) for doc in no_white]
    
    pre = [re.sub(pattern, '', i) for i in no_stopwords]
    
    
    
    too_small = ['' if (len(i.strip()) < 2) else i for i in pre]
    
    #return#(' '.join([word for word in simple_preprocess(str(doc)) 
            # if word not in stop_words]) != '' and 
    return[i.strip() != '' for i in too_small]
  
def process_in(texts):
  
    pattern = re.compile(r'[^A-Za-z\s]*')
    
    no_white = [i.strip() for i in texts]
    
    no_stopwords = [' '.join([word for word in simple_preprocess(str(doc)) if word not in stop_words]) for doc in no_white]
    
    pre = [re.sub(pattern, '', i) for i in no_stopwords]
    
    too_small = ['' if (len(i.strip()) < 2) else i for i in pre]
    
    #return#(' '.join([word for word in simple_preprocess(str(doc)) 
            # if word not in stop_words]) != '' and 
    # print([(i, j) for (i, j) in enumerate(too_small) if j == ''])        
    
    return[i for i in too_small if i.strip() != '']
             
 
def preprocess_for_LDA(comments):

  bools = check_stopwords(comments)
    
  sanitized_input = [j for (i,j) in zip(bools, comments) if i]
  
  return sanitized_input



if __name__ == '__main__':
  
  import pandas as pd
  
  reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())
  
  print("Valid Input")
  print(preprocess_for_LDA(reviews)[:2])
  
  print("Processed")
  print(process_in(reviews)[:2])
  

