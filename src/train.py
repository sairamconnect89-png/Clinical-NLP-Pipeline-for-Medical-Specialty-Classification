import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.config import *

def train():
    df = load_data()
    df['clean_text'] = df['text'].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=MAX_FEATURES)),
        ('clf', LinearSVC(class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, X_test, y_test
