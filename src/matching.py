import torch
from transformers import AutoModel, AutoTokenizer
from pyvi.ViTokenizer import tokenize as pyvi_tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from db_collection import get_ideas_key
from filtering_data import get_data_by_column, get_valid_idea_db
from db_connection import db_connection

PhobertTokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
SimPhobertModel = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")

class Embedding:
  def __init__(self) -> None:
    self.embedding = AutoModel.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
    self.tokenizer = AutoTokenizer.from_pretrained("VoVanPhuc/sup-SimCSE-VietNamese-phobert-base")
  
  def get_embedding(self, documents):
    if isinstance(documents, list):
      documents_tokenizer = [pyvi_tokenizer(docu) for docu in documents]
    else:
      documents_tokenizer = pyvi_tokenizer(documents)
    inputs = self.tokenizer(documents_tokenizer, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
      outputs = self.embedding(**inputs)

    # Use the mean pooling of the token embeddings as the sentence embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

class TopSimilarity:
  def __init__(self) -> None:
    self.sim_func = cosine_similarity

  def cal_cosine_sim(self, query_embedding, doc_embeddings):
    return self.sim_func(doc_embeddings, query_embedding)

  def search_indices(self, query_embedding, doc_embeddings, topk = 20):
    cos_similarities = self.cal_cosine_sim(doc_embeddings, query_embedding)

    # Get indices of nearest neighbors based on cosine similarity
    indices = np.argsort(cos_similarities.flatten())[::-1][:topk]
    result = [(idx, cos_similarities.flatten()[idx]) for idx in indices]
    return result # [(0, 0.8), (12, 0.78)]

def find_topk_idea(req: str, topk: int = 20):
  sim_func = TopSimilarity()
  embedding_model = Embedding()
  
  # read db into smaller batches (1000)
  db = get_ideas_key()

  # split db into smaller batches (1000)
  batch_size = 100

  # Calculate the total number of batches
  num_batches = len(db) // batch_size + 1
  print(num_batches)

  result = []
  # Loop over each batch
  for batch_num in range(num_batches):
    # Calculate the start and end indices of the current batch
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size if start_idx + batch_size < len(db) else len(db)
    # embedding db

    batch_db = db[start_idx:end_idx]

    # Extract values associated with the 'solution' key into a list
    solutions = [item['solution'] for item in batch_db]

    db_embedding = embedding_model.get_embedding(solutions)

    # embedding query
    query_embedding = embedding_model.get_embedding(req)
    tuple_idx_cosim = sim_func.search_indices(query_embedding, db_embedding)
    tuple_idx_cosim = [(batch_db[idx]['_id'], cosim) for idx, cosim in tuple_idx_cosim]
    
    # Concatenate the two lists
    concat_list = tuple_idx_cosim + result

    # Sort the concatenated list by the similarity value in descending order
    sorted_list = sorted(concat_list, key=lambda x: x[1], reverse=True)

    # Get the top 20 tuples (or fewer if the list has less than 20 items)
    topk_tuples = sorted_list[:topk]
    result = topk_tuples
    
  return result

def cal_sim(query, sent):
  sim_func = TopSimilarity()
  embedding_model = Embedding()
  query_emb = embedding_model.get_embedding(query)
  sent_emb = embedding_model.get_embedding(sent)
  return sim_func.cal_cosine_sim(query_emb, sent_emb)
 
def find_topk(req: str, topk: int = 20, idea_db: list = None):
  sim_func = TopSimilarity()
  embedding_model = Embedding()
  
  # read db into smaller batches (1000)
  db = get_valid_idea_db()[:1000]

  # split db into smaller batches (1000)
  batch_size = 100

  # Calculate the total number of batches
  num_batches = len(db) // batch_size

  result = []
  # Loop over each batch
  for batch_num in range(num_batches):
    # Calculate the start and end indices of the current batch
    start_idx = batch_num * batch_size
    end_idx = start_idx + batch_size
    # embedding db

    batch_db = db[start_idx:end_idx]

    db_embedding = embedding_model.get_embedding(batch_db)

    # embedding query
    query_embedding = embedding_model.get_embedding(req)
    tuple_idx_cosim = sim_func.search_indices(query_embedding, db_embedding)
    tuple_idx_cosim = [(start_idx + idx, cosim) for idx, cosim in tuple_idx_cosim]
    
    # Concatenate the two lists
    concat_list = tuple_idx_cosim + result

    # Sort the concatenated list by the similarity value in descending order
    sorted_list = sorted(concat_list, key=lambda x: x[1], reverse=True)

    # Get the top 20 tuples (or fewer if the list has less than 20 items)
    topk_tuples = sorted_list[:topk]
    result = topk_tuples
    print('------ ', batch_num)
  return result