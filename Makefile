.PHONY: setup data train api app clean

setup:
	pip install -r requirements.txt

# Generate synthetic data (no Kaggle account needed)
data:
	python scripts/generate_demo_data.py

# Or download real data: kaggle datasets download mlg-ulb/creditcardfraud -p data --unzip
train:
	python src/train.py

api:
	uvicorn api.main:app --reload --port 8000

app:
	python app.py

# Full pipeline from scratch (synthetic data)
demo: data train app

clean:
	rm -rf artifacts/ data/creditcard.csv __pycache__ src/__pycache__ api/__pycache__
