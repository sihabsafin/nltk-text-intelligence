import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.classify import NaiveBayesClassifier

def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]

def text_to_features(text):
    tokens = preprocess(text)
    return {word: True for word in tokens}

training_data = [
    ("I love this product", "positive"),
    ("This is a great experience", "positive"),
    ("I hate this service", "negative"),
    ("This is a bad product", "negative"),
]

train_set = [(text_to_features(text), label) for text, label in training_data]
classifier = NaiveBayesClassifier.train(train_set)

def extract_entities(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    tree = ne_chunk(tags)
    entities = []
    for chunk in tree:
        if hasattr(chunk, 'label'):
            entities.append((chunk.label(), ' '.join(c[0] for c in chunk)))
    return entities

st.title("🧠 AI Text Intelligence Dashboard (NLTK)")
user_text = st.text_area("Enter your text here:")

if st.button("Analyze"):
    features = text_to_features(user_text)
    sentiment = classifier.classify(features)
    entities = extract_entities(user_text)

    st.subheader("Sentiment")
    st.success(sentiment)

    st.subheader("Keywords")
    st.write(list(features.keys()))

    st.subheader("Named Entities")
    if entities:
        for ent in entities:
            st.write(ent)
    else:
        st.write("No entities found")
