import numpy as np
import json
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing import sequence
# from keras.src.optimizers.legacy.rmsprop import RMSprop
# from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing import sequence

# from tensorflow.keras.models import load_model, save_model

from keras.saving import load_model
from gibberish_detector import detector
from filtering_data import get_data_by_column
from entity import Idea, IdeaObj

# model_pkl_file='model/spam_filtering.pkl'
# model_json_file = "model/spam_filtering.json"
# model_weights_file = "model/weights.h5py"
# word2vec_path='model/word2vec.vie.model.bin'
gibberish_path='model/gibberish.vie.model'
filtering_model_path = 'model/saved_model.keras'
max_words = 1000
max_len = 150

class IdeaTokenize:
  def __init__(self) -> None:
    self.tokenizer = Tokenizer(num_words=max_words)

    dataset_vocab = get_data_by_column('sentence')
    self.tokenizer.fit_on_texts(dataset_vocab)

  def tokenize_text(self, text):
    text = [text]
    # self.tokenizer.fit_on_texts(text)
    text_seq = self.tokenizer.texts_to_sequences(text)
    # text_seq_pad_matrix = sequence.pad_sequences(text_seq, maxlen=max_len)
    text_seq_pad_matrix = pad_sequences(text_seq, maxlen=max_len)
    return text_seq_pad_matrix

class FilteringModel:
  def __init__(self) -> None:
    self.detector = detector.create_from_model(gibberish_path)
    self.lstm_model = load_model(filtering_model_path)
    self.tok = IdeaTokenize()
  
  def predict(self, idea_object: Idea):
    for form_input in idea_object.__dict__.values():
      is_gibberish_input = self.detector.is_gibberish(form_input)
      if (is_gibberish_input):
        print('==== GIBBERISH DETECT')
        return ("SPAM", idea_object)

    # filtering solution
    idea_solution = idea_object.solution
    idea_id = idea_object.id
    tok_solution = self.tok.tokenize_text(idea_solution)
    prediction = self.lstm_model.predict(tok_solution)
    label = "SPAM"
    if prediction > 0.7:
      label = "VALID"
    elif prediction > 0.4:
      label = "WARNING"
    print(prediction, f' -> {label} :: "{idea_solution[:30]}..."')
    result = {
      "label": label,
      "_id": idea_id
    }
    return result

# problem = 'khả năng kích hoạt enzyme'
# solution = """Một hệ thống phân tích sinh học cơ thể, trực quan số liệu về hồng cầu, hệ miễn dịch,
# xem bệnh nhân, đang dần hồi phục qua các lần tái khám"""
# solution = 'Chào bạn, mình là Nghi sinh viên trường Đại học Bách khoa.'
# idea = Idea(id='234', problem = problem, solution = solution)
# predict_model = FilteringModel()
# res = predict_model.predict(idea)

# def filter(idea: Idea):
#   predict_model = FilteringModel()
#   return predict_model.predict(idea)