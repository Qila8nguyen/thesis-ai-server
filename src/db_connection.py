from pymongo import MongoClient, collection
import os
# from bson import json_util
from typing import Union
from dotenv import load_dotenv

class MongoDBClass:
  def __init__(
    self,
    database: str,
    url: str = '',
    host: str = 'localhost',
    port: Union[str, int] = '27017',
    user: str = '',
    password: str = '',
    ):

    self.url = url
    self.database = database
    self.host = host
    self.port = port
    self.user = user
    self.password = password

    if not url:
      self._client = MongoClient(self.url)
    else:
      self._client = MongoClient(
          f'{self.user}{self.password}@{self.host}:{self.port}/?ssl=true&ssl_ca_certs=rds-combined-ca-bundle.pem&replicaSet=rs0&readPreference=secondaryPreferred&retryWrites=false'
      )
    self._db = self._client[self.database]
    # self._collection = self._db.sample_collection

  def close(self):
    self._client.close()

  def insert_one(self, key: str, value: str):
    self._collection.insert_one({key: value})

def connect_get_db():
  load_dotenv()
  mongodb_url = os.getenv('MONGODB_URL')
  db_name = os.getenv('MONGODB_DATABASE')
  client = MongoClient(mongodb_url)
  return client.get_database(db_name)

class DBConnection(object):
  def __new__(self):
    if not hasattr(self, 'instance'):
      self.instance = super(DBConnection, self).__new__(self)
    return self.instance
  
  def connect_db(self) -> MongoClient:
    load_dotenv()
    CONNECTION_STRING = os.getenv("MONGODB_URL")
    client = MongoClient(CONNECTION_STRING)
    return client.get_database("share-your-idea")
  
  def get_collection(self, collection_name) -> collection.Collection:
    return self.instance.get_collection(collection_name)
  
  def close(self):
    self.instance.close()

db_connection = DBConnection()
db_connection.instance = db_connection.connect_db()

