from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from torch.cuda import is_available as has_cuda
import pandas as pd

def TexRank(comments):
  """
  Converts from a list to a graph, where each element in the list is a node, and
  edges between nodes are the cosine similarity between the sentence embedding of
  each element. Then, run pagerank on this graph to find an ordering for the initial
  list
  """
  
  if (len(comments) == 1):
    return({'Comment':comments, 'Score':1 })
  
  # Load a pre-trained model
  
  device = 'cuda' if has_cuda() else 'cpu'
  
  embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
  
  
  embeddings = embedding_model.encode(list(comments))
  
  similarity_matrix = cosine_similarity(embeddings)
  
  # normalise columns. massively increases performance.
  similarity_matrix = similarity_matrix/similarity_matrix.sum(axis=0)

  G = nx.Graph()

  for i in range(len(comments)):
      for j in range(i + 1, len(comments)):
          G.add_edge(i, j, weight=similarity_matrix[i, j])
          
  

  
  pagerank_scores = nx.pagerank(G, weight='weight', max_iter=10000)
  
  top_comments = pd.DataFrame({"Comment": comments,
                               "Score": [pagerank_scores[i] for i in range(len(pagerank_scores))]}
                               ).sort_values(by='Score', ascending=False)
                               
  return(top_comments)       
