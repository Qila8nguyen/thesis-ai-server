from tensorflow.keras.models import load_model

filtering_model_path = 'model/saved_model.keras'
lstm_model = load_model(filtering_model_path)