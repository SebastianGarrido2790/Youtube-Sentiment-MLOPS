# Create the top-level directories
mkdir -p data/external data/interim data/processed data/raw
mkdir -p models notebooks references reports/figures src/data src/features src/models src/visualization

# Create the top-level files
touch .env .gitignore README.md requirements.txt LICENSE

# Create the Python scripts within src
touch src/__init__.py
touch src/data/make_dataset.py
touch src/features/feature_engineering.py
touch src/models/train_model.py
touch src/models/predict_model.py
touch src/visualization/visualize.py