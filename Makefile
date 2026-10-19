.PHONY: install data eda train test lint serve docker smoke k8s-deploy monitor

install:
	pip install -r requirements-dev.txt
data:
	python scripts/download_data.py
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest
lint:
	flake8 src tests scripts
serve:
	uvicorn src.api:app --reload --port 8000
docker smoke:
	./scripts/smoke_test.sh
k8s-deploy:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
monitor:
	docker compose up --build
