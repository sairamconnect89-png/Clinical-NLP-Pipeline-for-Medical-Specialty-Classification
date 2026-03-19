import pandas as pd
from src.config import DATA_PATH, TOP_LABELS

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df[['transcription', 'medical_specialty']]
    df.dropna(inplace=True)

    df.rename(columns={
        'transcription': 'text',
        'medical_specialty': 'label'
    }, inplace=True)

    top_labels = df['label'].value_counts().nlargest(TOP_LABELS).index
    df = df[df['label'].isin(top_labels)]

    return df
