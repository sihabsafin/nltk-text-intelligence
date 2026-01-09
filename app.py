import streamlit as st
import nltk
import json
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
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng")
    nltk.download("maxent_ne_chunker")
    nltk.download("maxent_ne_chunker_tab")
    nltk.download("words")

download_nltk()

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(
    page_title="AI Text Intelligence",
    page_icon="🧠",
    layout="centered"
)

# ============================
# UI STYLE (UNCHANGED)
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
# SESSION STATE (HISTORY)
# ============================
if "history" not in st.session_state:
    st.session_state.history = []

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

            # Sentiment + confidence
            prob_dist = classifier.prob_classify(features)
            sentiment = prob_dist.max()
            confidence = round(prob_dist.prob(sentiment) * 100, 2)

            entities = extract_entities(user_text)
            keywords = list(features.keys())

        # Save history
        result = {
            "text": user_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "keywords": keywords,
            "entities": entities
        }
        st.session_state.history.append(result)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Sentiment")
            st.success(f"{sentiment.capitalize()} ({confidence}%)")

        with col2:
            st.subheader("🔑 Keywords")
            st.write(keywords)

        st.subheader("🏷️ Named Entities")
        if entities:
            for label, value in entities:
                st.write(f"**{label}** → {value}")
        else:
            st.info("No named entities found")

        # ============================
        # DOWNLOAD REPORT (JSON)
        # ============================
        st.download_button(
            label="📥 Download Analysis Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name="text_analysis_report.json",
            mime="application/json"
        )

# ============================
# HISTORY & COMPARISON
# ============================
if st.session_state.history:
    st.divider()
    st.subheader("🕘 Analysis History")

    for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
        st.markdown(f"""
**#{i} Text:** {item['text']}  
- Sentiment: **{item['sentiment']} ({item['confidence']}%)**  
- Keywords: `{", ".join(item['keywords'])}`
""")

st.divider()
st.caption("Portfolio Project • NLP Foundations • Streamlit Cloud Ready")
