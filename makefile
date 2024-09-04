# Mengumpulkan dan mempersiapkan data
data:
	python scripts/data_prep.py

# Melatih model
train: data
	python scripts/train_model.py

# Mengevaluasi model
evaluate: train
	python scripts/evaluate_model.py

# Simpan model
deploy: train
	@echo "Model sudah dilatih dan disimpan sebagai models/model.pkl"