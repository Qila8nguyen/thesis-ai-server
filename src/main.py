from fastapi import FastAPI, HTTPException
from typing import List, Tuple
import os
from route import filter, topk_search, topk_idea_search
from db_connection import db_connection
from entity import Idea, Requirement
from bson import ObjectId
from app_setting import app

## uvicorn main:app --reload
## pyenv -> 3.10.14
# pip freeze > requirements.txt

@app.get("/spam_filtering")
def spam_filtering(idea: Idea):
  return filter(idea)

@app.get("/topk")
def search_topk_similarity(req: Requirement):
  try:
    query = f'{req.problem}. {req.acceptance_criteria}'
    topk_id = topk_idea_search(query)
    result = {
      "_id": req.id,
      "rank": topk_id
    }
    return result
  except Exception as e:
    print(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/topk_internal", response_model=List[Tuple[str, float]])
def search_topk_similarity(req: Requirement):
  try:
    query = f'{req.problem}. {req.acceptance_criteria}'
    return topk_search(query)
  except Exception as e:
    print(f"Unexpected error: {e}")
    raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/ideas")
def get_ideas():
  idea_collection = db_connection.get_collection("idea")
  # Retrieve documents from the collection
  documents = idea_collection.find()
  # Convert documents to a readable Python format
  readable_data = []
  for doc in documents:
      readable_doc = {}
      for key, value in doc.items():
          # Convert ObjectId to string for better readability
          if isinstance(value, ObjectId):
              value = str(value)
          readable_doc[key] = value
      readable_data.append(readable_doc)

  # Print the readable data
  for doc in readable_data:
      print(doc)
  return 1

# @app.on_event("startup")
# async def startup():
#   await connect_get_db()

# @app.on_event("shutdown")
# async def shutdown():
#     await db_connection.disconnect()

# if __name__ == '__main__':
#   # connect db
#   mongodb_url = os.getenv('MONGODB_URL')
#   db_name = os.getenv('MONGODB_DATABASE')
#   print(mongodb_url, db_name)
#   db.MongoDBClass(db_name, mongodb_url)