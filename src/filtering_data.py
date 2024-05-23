import pandas as pd

def get_dataframe():
  # abs_path = '../'
  invalid_1 = pd.read_csv("data/random_str.csv", delimiter=',')
  invalid_0 = pd.read_csv("data/vie/spam.csv", delimiter=';')
  semantic = pd.read_csv("data/vie/valid_idea.csv", delimiter=';', usecols=['sentence', 'spam'])

  invalid_0.head()
  invalid_1.head()
  semantic.head()

  frames = [invalid_0, semantic, invalid_1]
  
  df = pd.concat(frames)

  return df

def get_data_by_column(col: str = None, df = None):
  if df:   
    return df[col] if col else df
  df = get_dataframe()
  return df[col] if col else df

def get_valid_idea_db():
  # abs_path = '../'
  abs_path = ''
  semantic = pd.read_csv(abs_path + "data/vie/valid_idea.csv", delimiter=';', usecols=['sentence', 'spam'])
  semantic = semantic['sentence'].to_list()
  
  return semantic