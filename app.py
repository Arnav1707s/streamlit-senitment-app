import streamlit as st
import pandas as pd
import re
import nltk
import os
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_dir)
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# ✅ Preprocessing Functions
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(tokens)
    words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
        if word not in stop_words and len(word) > 2
    ]
    return ' '.join(words)

# ✅ Improved Star Rating Function
def get_star_rating(sentiment, confidence):
    if sentiment == 1:  # Positive
        if confidence >= 0.9:
            return "⭐⭐⭐⭐⭐"
        elif confidence >= 0.75:
            return "⭐⭐⭐⭐"
        else:
            return "⭐⭐⭐"
    elif sentiment == 0:  # Neutral
        if confidence >= 0.8:
            return "⭐⭐⭐"
        else:
            return "⭐⭐"
    else:  # Negative (force low stars)
        if confidence >= 0.9:
            return "⭐⭐"
        else:
            return "⭐"

# ✅ Emoji Feedback Function
def get_emoji_feedback(sentiment, confidence):
    if sentiment == 1:
        if confidence >= 0.9:
            return "😍 Extremely Happy"
        elif confidence >= 0.75:
            return "😊 Happy"
        else:
            return "🙂 Slightly Positive"
    elif sentiment == 0:
        if confidence >= 0.8:
            return "😐 Neutral"
        else:
            return "🤔 Slightly Neutral"
    else:
        if confidence >= 0.9:
            return "😡 Very Dissatisfied"
        elif confidence >= 0.75:
            return "😠 Angry"
        else:
            return "😞 Slightly Negative"

# ✅ Model Training (If Not Already Trained)
model_path = 'sentiment_model_fixed.pkl'
if not os.path.exists(model_path):
    st.info("🔨 Training model using your IMDB CSV with Correct Mapping...")

    # ✅ Load your IMDB dataset
    data = pd.read_csv("imdb_reviews.csv")
    
    # ✅ Correct Sentiment Mapping:
    data['sentiment'] = data['sentiment'].map({0: -1, 1: 1})

    # ✅ Inject Neutral Samples (Optional)
    neutral_data = pd.DataFrame({
        'review': [
            "It was okay",
            "Average movie",
            "Nothing special or bad",
            "It was just fine, not great, not bad",
            "Neutral experience overall",
            "I neither liked nor disliked it"
        ],
        'sentiment': [0, 0, 0, 0, 0, 0]
    })

    data = pd.concat([data, neutral_data], ignore_index=True)

    # ✅ Preprocessing & Training
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

# ✅ Load Model
model = joblib.load(model_path)

# ✅ Streamlit App UI
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

        stars = get_star_rating(sentiment, confidence)  # ✅ Fixed (added sentiment)
        emoji = get_emoji_feedback(sentiment, confidence)

        sentiment_label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]

        st.success("🎯 **Prediction Result:**")
        st.markdown(f"**Sentiment:** `{sentiment_label}`")
        st.markdown(f"**Confidence:** `{confidence * 100:.2f}%`")
        st.markdown(f"**Star Rating:** {stars}")
        st.markdown(f"**Emoji Feedback:** {emoji}")

st.markdown("---")
st.caption("🚀 Built with Streamlit, TF-IDF, Logistic Regression & NLP")

