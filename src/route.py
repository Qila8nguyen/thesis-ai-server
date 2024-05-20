from entity import Idea
from filtering_model import FilteringModel
from matching import find_topk, find_topk_idea
import time
from filtering_data import get_db
from db_collection import get_ideas, get_idea_by_id

db = get_db()['sentence'].to_list()[:3000]
ideas = get_ideas()

def filter(idea: Idea):
  predict_model = FilteringModel()
  return predict_model.predict(idea)

def topk_search(req: str):
  # Calculate the elapsed time
  start_time = time.time()
  
  topk = find_topk(req)
  #---
  end_time = time.time()
  elapsed_time = end_time - start_time
  for idc, sim in topk:
    print(f'Idx: {idc} => {db[idc]} \n cossim = {sim}')

  print(f"Elapsed time: {elapsed_time:.6f} seconds")

  return topk

def topk_idea_search(req: str):
  # Calculate the elapsed time
  start_time = time.time()
  
  # topk = find_topk(req)
  topk = find_topk_idea(req)
  #---
  end_time = time.time()
  elapsed_time = end_time - start_time
  for id_str, sim in topk[:2]:
    idea = get_idea_by_id(id_str)
    if idea:
      print(f'Idx: {id_str} => {idea["problem"]} \n >>> cossim = {sim}')
    else:
      print('Not found')

  print(f"Elapsed time: {elapsed_time:.6f} seconds")
  result = [id for id, cos_sim in topk]
  return result