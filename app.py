import streamlit as st
import pandas as pd
import re
import os
import nltk

# ✅ Define and register custom NLTK data directory early
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# ✅ Ensure all required NLTK resources are available
resources = {
    'punkt': 'tokenizers/punkt',
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
}
for resource, path in resources.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# ✅ NLP imports
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, TreebankWordTokenizer

# ✅ ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# ✅ POS tagging helper with fallback
def safe_pos_tag(tokens):
    try:
        return pos_tag(tokens)
    except LookupError:
        nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
        return pos_tag(tokens)

# ✅ Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    pos_tags = safe_pos_tag(tokens)

    def get_wordnet_pos_simple(tag):
        return {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}.get(tag[0], 'n')

    words = [lemmatizer.lemmatize(word, get_wordnet_pos_simple(tag))
             for word, tag in pos_tags if word not in stop_words]

    return ' '.join(words)

# ✅ Star rating
def get_star_rating(sentiment, confidence):
    if sentiment == 1:
        return "⭐⭐⭐⭐⭐" if confidence >= 0.9 else "⭐⭐⭐⭐" if confidence >= 0.75 else "⭐⭐⭐"
    elif sentiment == 0:
        return "⭐⭐⭐" if confidence >= 0.8 else "⭐⭐"
    else:
        return "⭐⭐" if confidence >= 0.9 else "⭐"

# ✅ Emoji feedback
def get_emoji_feedback(sentiment, confidence):
    if sentiment == 1:
        return "😍 Extremely Happy" if confidence >= 0.9 else "😊 Happy" if confidence >= 0.75 else "🙂 Slightly Positive"
    elif sentiment == 0:
        return "😐 Neutral" if confidence >= 0.8 else "🤔 Slightly Neutral"
    else:
        return "😡 Very Dissatisfied" if confidence >= 0.9 else "😠 Angry" if confidence >= 0.75 else "😞 Slightly Negative"

# ✅ Model training or loading
model_path = 'sentiment_model_fixed.pkl'
if not os.path.exists(model_path):
    st.info("🔨 Training model using your IMDB CSV with Correct Mapping...")

    data = pd.read_csv("imdb_reviews.csv")
    data['sentiment'] = data['sentiment'].map({0: -1, 1: 1})

    neutral_data = pd.DataFrame({
        'review': [
            "It was okay", "Average movie", "Nothing special or bad",
            "It was just fine, not great, not bad", "Neutral experience overall", "I neither liked nor disliked it"
        ],
        'sentiment': [0] * 6
    })

    data = pd.concat([data, neutral_data], ignore_index=True)
    data['review'] = data['review'].astype(str).apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        data['review'], data['sentiment'], test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,
            stop_words='english'
        )),
        ('clf', LogisticRegression(
            max_iter=1200,
            C=1.5,
            multi_class='multinomial',
            solver='lbfgs',
            class_weight='balanced',
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("Model Accuracy Report:\n", classification_report(y_test, y_pred))

    joblib.dump(pipeline, model_path)
    st.success("✅ Model trained and saved successfully!")
else:
    print("✅ Model already trained. Loading...")

model = joblib.load(model_path)

# ✅ Streamlit UI
st.set_page_config(page_title="3-Class Sentiment Classifier", page_icon="💬", layout="centered")
st.title("💬 Multi-Class Sentiment Classifier (Positive, Neutral, Negative)")
st.subheader("🔍 Analyze Sentiment with Star Rating & Emojis")

review = st.text_area("📝 Enter Your Review:")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        processed_review = preprocess_text(review)
        sentiment = model.predict([processed_review])[0]
        confidence = model.predict_proba([processed_review]).max()

        stars = get_star_rating(sentiment, confidence)
        emoji = get_emoji_feedback(sentiment, confidence)

        sentiment_label = {1: "Positive", 0: "Neutral", -1: "Negative"}.get(sentiment, "Unknown")

        st.success("🎯 **Prediction Result:**")
        st.markdown(f"**Sentiment:** `{sentiment_label}`")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        st.markdown(f"**Star Rating:** {stars}")
        st.markdown(f"**Emoji Feedback:** {emoji}")

st.markdown("---")
st.caption("🚀 Built with Streamlit, TF-IDF, Logistic Regression & NLP")
