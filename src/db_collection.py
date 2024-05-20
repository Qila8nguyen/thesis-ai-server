from db_connection import db_connection
from bson import ObjectId

def get_ideas():
  idea_collection = db_connection.get_collection("idea")
  # Retrieve documents from the collection
  documents = idea_collection.find()
  # Convert documents to a readable Python format
  idea_list = []
  for doc in documents:
      readable_doc = {}
      for key, value in doc.items():
          # Convert ObjectId to string for better readability
          if isinstance(value, ObjectId):
              value = str(value)
          readable_doc[key] = value
      idea_list.append(readable_doc)

  # Print the readable data
  # for doc in idea_list:
      # print(doc)
  
  return idea_list

def get_ideas_key():
  # just get _id, problem, solution
  idea_collection = db_connection.get_collection("idea")
  # Retrieve documents from the collection
  documents = idea_collection.find()
  # Convert documents to a readable Python format
  idea_list = []
  for doc in documents:
    keys = ['problem', 'solution']
    readable_doc = {'solution': ''}
    for key, value in doc.items():
      # Convert ObjectId to string for better readability
      if isinstance(value, ObjectId):
        value = str(value)
        readable_doc[key] = value
      if key in keys:
        readable_doc['solution'] += value
    idea_list.append(readable_doc) # [{_id: '...', solution: '...'}]
  return idea_list

def get_idea_by_id(_id: str):
  # Define your condition/query
  objectid = ObjectId(_id)
  query = {"_id": objectid}  # Example condition, replace with your actual condition

  # Find documents matching the condition
  results = db_connection.get_collection("idea").find_one(query)
  return results
