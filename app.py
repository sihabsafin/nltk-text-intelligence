import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from nltk.classify import NaiveBayesClassifier

# ============================
# NLTK DATA (PYTHON 3.13 SAFE)
# ============================
@st.cache_resource
def download_nltk():
    nltk.download("punkt")
    nltk.download("punkt_tab")   # 🔥 REQUIRED (NEW)
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("maxent_ne_chunker")
    nltk.download("words")

download_nltk()

# ============================
# PAGE CONFIG (MODERN)
# ============================
st.set_page_config(
    page_title="AI Text Intelligence",
    page_icon="🧠",
    layout="centered"
)

# ============================
# PREMIUM UI STYLE
# ============================
st.markdown("""
<style>
body { background-color: #0e1117; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ============================
# NLP PIPELINE
# ============================
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(w) for w in tokens]

def text_to_features(text):
    tokens = preprocess(text)
    return {word: True for word in tokens}

# ============================
# TRAIN SENTIMENT MODEL
# ============================
training_data = [
    ("I love this product", "positive"),
    ("This is a great experience", "positive"),
    ("Amazing service and support", "positive"),
    ("I hate this service", "negative"),
    ("This is a bad product", "negative"),
    ("Very poor experience", "negative"),
]

train_set = [(text_to_features(text), label) for text, label in training_data]
classifier = NaiveBayesClassifier.train(train_set)

# ============================
# ENTITY EXTRACTION
# ============================
def extract_entities(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    tree = ne_chunk(tags)

    entities = []
    for chunk in tree:
        if hasattr(chunk, "label"):
            entities.append((chunk.label(), " ".join(c[0] for c in chunk)))
    return entities

# ============================
# UI
# ============================
st.title("🧠 AI Text Intelligence Dashboard")
st.caption("Sentiment • Keywords • Named Entities | Built with NLTK")

st.divider()

user_text = st.text_area(
    "Enter text for analysis",
    placeholder="Example: Elon Musk founded SpaceX in America and I love this company.",
    height=150
)

if st.button("🚀 Analyze Text", use_container_width=True):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing text..."):
            features = text_to_features(user_text)
            sentiment = classifier.classify(features)
            entities = extract_entities(user_text)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment")
            st.success("Positive" if sentiment == "positive" else "Negative")

        with col2:
            st.subheader("🔑 Keywords")
            st.write(list(features.keys()))

        st.subheader("🏷️ Named Entities")
        if entities:
            for label, value in entities:
                st.write(f"**{label}** → {value}")
        else:
            st.info("No named entities found")

st.divider()
st.caption("Portfolio Project • NLP Foundations • Streamlit Cloud Ready")
