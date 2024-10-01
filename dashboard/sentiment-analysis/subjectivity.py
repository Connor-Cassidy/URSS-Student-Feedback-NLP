"""
This file contains a function which takes in a pandas dataframe of the form

 |   col_1  |   col_2  | ... |   col_P  |
 ----------------------------------------
 | 1_text_1 | 1_text_2 | ... | 1_text_P |
 | 2_text_2 | 2_text_2 | ... | 2_text_P |
 |    :     |    :     |  :  |   :      |
 | N_text_1 | N_text_2 | ... | N_text_P |
 
and maps each i_text_j to a subjectivity score i_subjectivity_j. To classify the
subjectivity of each text, we use 


This subjectivity score is a value from 0 to 1, with score close to 1 indicating
highly subjective text, and values close to 0 highly objective. The output 
dataframe is thus of the form.

 |    col_1      |     col_2     | ... |     col_P     |
 -------------------------------------------------------
 | 1_subjectivity_1 | 1_subjectivity_2 | ... | 1_subjectivity_P |
 | 2_subjectivity_2 | 2_subjectivity_2 | ... | 2_subjectivity_P |
 |       :       |       :       |  :  |       :       |
 | N_subjectivity_1 | N_subjectivity_2 | ... | N_subjectivity_P |
 
Each i_text_j is expected to be one review, with each column either representing
a question or topic
 
"""





from textblob import TextBlob

def extract_subjectivity(input_dataframe):
    # First, ensure the input is of correct form, and coerce to string if not.
    if (input_dataframe.map(lambda x: not isinstance(x, str)).any().any()):
        warn(f"Some cell in {input_dataframe} is not a string, attempting to coerce to string")
        input_dataframe = input_dataframe.applymap(str)
    
    # Apply TextBlob to calculate subjectivity for each cell
    subjectivity_unscaled = input_dataframe.map(lambda text: TextBlob(text).sentiment.subjectivity).reset_index(drop=True)
    
    # Scale from (0,1) to (-1,1) to match sentiment
    subjectivity = subjectivity_unscaled.map(lambda x: 2*x - 1)
    
    
    return subjectivity




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
             
  combined_review_df = pd.DataFrame({
    "social media posts": sentences,
    "amazon reviews": reviews
    })
  
  print(extract_subjectivity(combined_review_df).iat[0,0])
