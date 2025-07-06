import nltk
import os

# Create a safe local nltk_data folder
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Append this directory to NLTK's data path
nltk.data.path.append(nltk_data_dir)

# Download necessary resources
nltk.download("punkt", download_dir=nltk_data_dir)
nltk.download("stopwords", download_dir=nltk_data_dir)
nltk.download("wordnet", download_dir=nltk_data_dir)
nltk.download("averaged_perceptron_tagger", download_dir=nltk_data_dir)

print("✅ All NLTK data downloaded to:", nltk_data_dir)
