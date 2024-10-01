"""
This file contains a function which takes in a list of the form

[  1_text,  2_text, ..., N_text ]

alongside a model, a string which is one of

facebook/bart-large-cnn

google-t5/t5-small

google-t5/t5-large

google-t5/t5-3b

google-t5/t5-11b

google/umt5-small ? 

google/umt5-xxl ? 

"""

import pandas as pd
import torch
import numpy as np

import sys
sys.path.insert(0, './sentiment-analysis/')
sys.path.insert(0, 'sentiment-analysis/')
from positivity import extract_positivity


implemented_models = (
  "facebook/bart-large-cnn",
  # "google-t5/t5-small",
  "google-t5/t5-large",
  # "google-t5/t5-3b",
  # "google-t5/t5-11b"
  # "google/umt5-small",
  # "google/umt5-xxl"
)

# If text is too large, recursively split and summarize each split

def generate_summary(text, tokenizer, model, 
                     min_length=30, max_length=150, do_sample=False, temperature=1, length_penalty = 2, num_beams = 8, device_type = "cpu", no_repeat_ngram_size=3):
    
    
    print(f"{len(text)}")
    
    
    max_input_length = 4_000
    
    # if (len(text) > max_input_length):
    #     
    #     review_end_token = "|"
    #     
    #     reviews = text.split(review_end_token)
    #     
    #     # Remove empty reviews
    #     reviews = [review for review in reviews if review.strip()]
    # 
    #     # Calculate the index to split the reviews list in half
    #     half_index = len(reviews) // 2
    # 
    #     # Recombine the reviews to form two halves
    #     first_half = review_end_token.join(reviews[:half_index]) + review_end_token
    #     
    #     second_half = review_end_token.join(reviews[half_index:]) + review_end_token
    #     
    #     first_summary = generate_summary(first_half,
    #                                     tokenizer, model,
    #                                     min_length, max_length,
    #                                     do_sample,
    #                                     temperature, length_penalty, num_beams, 
    #                                     device_type, no_repeat_ngram_size)
    #     
    #     
    #     second_summary = generate_summary(second_half,
    #                                     tokenizer, model,
    #                                     min_length, max_length,
    #                                     do_sample,
    #                                     temperature, length_penalty, num_beams, 
    #                                     device_type, no_repeat_ngram_size)
    #     
    #     final_summary = generate_summary(first_summary + second_summary,
    #                                     tokenizer, model,
    #                                     min_length, max_length,
    #                                     do_sample,
    #                                     temperature, length_penalty, num_beams, 
    #                                     device_type, no_repeat_ngram_size)
    #     
    #     return(final_summary)
      
    if (len(text) > max_input_length):
      
      arr = np.array(list(text))
    
      # Calculate the midpoint
      mid = len(arr) // 2
    
      # Split the array into two halves
      first_half = ''.join(arr[:mid])
      second_half = ''.join(arr[mid:])
      
      first_summary = generate_summary(first_half,
                                        tokenizer, model,
                                        min_length, max_length,
                                        do_sample,
                                        temperature, length_penalty, num_beams, 
                                        device_type, no_repeat_ngram_size)
      
      second_summary = generate_summary(second_half,
                                        tokenizer, model,
                                        min_length, max_length,
                                        do_sample,
                                        temperature, length_penalty, num_beams, 
                                        device_type, no_repeat_ngram_size)
  
      
      final_summary = generate_summary(first_summary + second_summary,
                                        tokenizer, model,
                                        min_length, max_length,
                                        do_sample,
                                        temperature, length_penalty, num_beams, 
                                        device_type, no_repeat_ngram_size)
      return final_summary
      
    else:
        

        text = text.replace('|', '')
        inputs = tokenizer(text, return_tensors="pt", max_length=max_input_length, truncation=True).to(device_type)



        summary_ids = model.generate(inputs["input_ids"], max_length=max_length, 
                                     min_length=min_length, length_penalty=length_penalty, num_beams=num_beams, early_stopping=True,
                                     no_repeat_ngram_size=no_repeat_ngram_size,
                                        temperature=temperature)


        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        if summary is None:
          print(text)
        
        return(summary)

def combine_text(input_list):
  # Combine all texts into a single string
  combined_text = ""
  for text in input_list:
    combined_text += f"{text} |"
  return(combined_text)




def abstractive_summary(input_list,model_name="facebook/bart-large-cnn", min_length = 30, max_length= 150, do_sample=False, temperature=1, length_penalty=2, num_beams=8, no_repeat_ngram_size=3):
  
  if (model_name not in implemented_models):
    raise NotImplementedError("The model {model} has not been implemented")
  
  
  
  # if (not torch.cuda.is_available()):
  #   raise NotImplementedError("WHY")
  # 
  # device_type = "cuda"
  device_type = "cpu"
  # device = torch.device(device_type)
  
  
  if model_name == "facebook/bart-large-cnn":
    
    from transformers import BartForConditionalGeneration, BartTokenizer
    
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device_type)
    tokenizer = BartTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    
  elif model_name in ["google-t5/t5-small", "google-t5/t5-large", "google-t5/t5-3b", "google-t5/t5-11b"]:
    
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    model = T5ForConditionalGeneration.from_pretrained(model_name,# device_map = 'auto'
                                                       ).to("device_type")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    
  combined_total = combine_text(input_list)
    
  # print(f"TOTAL {len(combined_total)}")  

                            
  summary = generate_summary(combined_total, 
                                   tokenizer, model, 
                                   min_length, max_length,
                                   do_sample, temperature, 
                                   length_penalty, num_beams,
                                  device_type,
                                  no_repeat_ngram_size
                                   )
                                   
                                   
  return(summary)
          
          


def summarize_by_sentiment(input_list, model_name="facebook/bart-large-cnn", pos_threshold=0.1, neg_threshold=-0.1, 
                           min_length = 30, max_length= 150, do_sample=False, temperature=1, length_penalty=2, num_beams=8, no_repeat_ngram_size=3):
  
  
  reviews_with_sentiment = extract_positivity(pd.DataFrame(input_list))
  
  reviews_with_sentiment["Review"] = pd.Series(input_list).reset_index(drop=True)
  
  # Extract reviews where compound score is above the pos threshhold
  pos_reviews = reviews_with_sentiment[reviews_with_sentiment['compound'] > pos_threshold]['Review'].tolist()
  
  # Extract reviews where compound score is below the neg threshold
  neg_reviews = reviews_with_sentiment[reviews_with_sentiment['compound'] < neg_threshold]['Review'].tolist()
  
  
  
  reviews_dict = {
      "pos": pos_reviews,
      "neg": neg_reviews,
      "total": input_list
  }
  
  summaries = {key: abstractive_summary(reviews_dict[key], 
                                        model_name, 
                                        min_length, max_length, 
                                        do_sample, temperature, 
                                        length_penalty, num_beams, 
                                        no_repeat_ngram_size)
               for key in reviews_dict}
  
  return summaries
  
          
if __name__ == "__main__":
    
    
    reviews = ["""My lovely Pat has one of the GREAT voices of her generation. 
             I have listened to this CD for YEARS and I still LOVE IT. When I'm in a good mood 
             it makes me feel better. A bad mood just evaporates like sugar in the rain. 
             This CD just oozes LIFE. Vocals are jusat STUUNNING and lyrics just kill. 
             One of life's hidden gems. This is a desert isle CD in my book. 
             Why she never made it big is just beyond me. Everytime I play this, 
             no matter black, white, young, old, male, female EVERYBODY says one thing 
             'Who was that singing ?'""",
               
             """Despite the fact that I have only played a small portion of the game, 
             the music I heard (plus the connection to Chrono Trigger which was great as well) 
             led me to purchase the soundtrack, and it remains one of my favorite albums. 
             There is an incredible mix of fun, epic, and emotional songs. 
             Those sad and beautiful tracks I especially like, as there's not 
             too many of those kinds of songs in my other video game soundtracks. 
             I must admit that one of the songs (Life-A Distant Promise) has brought 
             tears to my eyes on many occasions.My one complaint about this soundtrack
             is that they use guitar fretting effects in many of the songs,
             which I find distracting. But even if those weren't included 
             I would still consider the collection worth it.""",
               
             """I bought this charger in Jul 2003 and it worked OK for a while. 
             The design is nice and convenient. However, after about a year, 
             the batteries would not hold a charge. Might as well just get 
             alkaline disposables, or look elsewhere for a charger that comes 
             with batteries that have better staying power.""",
               
             """Check out Maha Energy's website. Their Powerex MH-C204F charger
             works in 100 minutes for rapid charge, with option for slower 
             charge (better for batteries). And they have 2200 mAh batteries.""",
               
             """Reviewed quite a bit of the combo players and was hesitant due 
             to unfavorable reviews and size of machines. I am weaning off my 
             VHS collection, but don't want to replace them with DVD's. 
             This unit is well built, easy to setup and resolution and special
             effects (no progressive scan for HDTV owners) suitable for many 
             people looking for a versatile product.Cons- No universal remote.""",
             
             """I also began having the incorrect disc problems that I've read about on here. 
             The VCR still works, but hte DVD side is useless. I understand that DVD players 
             sometimes just quit on you, but after not even one year? To me that's a sign on 
             bad quality. I'm giving up JVC after this as well. I'm sticking to Sony or giving another brand a shot.""",
             
             """I love the style of this, but after a couple years, the DVD is giving me problems. 
             It doesn't even work anymore and I use my broken PS2 Now. I wouldn't recommend this, 
             I'm just going to upgrade to a recorder now. I wish it would work but I guess 
             i'm giving up on JVC. I really did like this one... before it stopped working. 
             The dvd player gave me problems probably after a year of having it."""
             ]
             
    reviews = list(pd.read_csv("../education data/course_data_clean.csv")["reviews"].dropna())[0:50]       
    # print(summarize_by_sentiment(reviews, model_name="google-t5/t5-large", no_repeat_ngram_size=2, num_beams=8))
  
