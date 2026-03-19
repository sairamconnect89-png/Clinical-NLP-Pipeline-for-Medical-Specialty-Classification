import joblib
from src.config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(text):
    return model.predict([text])[0]
