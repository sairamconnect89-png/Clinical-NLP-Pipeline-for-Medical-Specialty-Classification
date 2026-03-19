# Clinical NLP Production System

## Features
- Text classification for medical specialties
- Streamlit dashboard
- Dockerized deployment

## Run Locally
pip install -r requirements.txt

python -m nltk.downloader stopwords

python src/train.py

streamlit run app/app.py

## Run with Docker
docker-compose up --build
