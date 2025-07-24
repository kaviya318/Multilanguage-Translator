import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, mean_absolute_error, r2_score, classification_report,
    accuracy_score, f1_score, mean_squared_error
)
import speech_recognition as sr
from gtts import gTTS
import plotly.express as px

# Set Streamlit page config
st.set_page_config(page_title="Multilingual Translator", page_icon="🌐", layout="wide")

# Load design.html (your background or animation)
with open("design.html", "r", encoding="utf-8") as f:
    design_html = f.read()
st.markdown(design_html, unsafe_allow_html=True)

# Initialize session state
for key in ["is_recording", "input_text", "final_sentence"]:
    if key not in st.session_state:
        st.session_state[key] = False if key == "is_recording" else ""

# Utility Functions
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)

def text_to_audio(text, lang):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save("translated_audio.mp3")
        return "translated_audio.mp3"
    except:
        return None

def record_audio():
    r = sr.Recognizer()
    st.info("🎤 Recording... Speak now!")
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

def generate_dummy(num_samples=100):
    return np.random.rand(num_samples, 768), np.random.choice([0, 1, 2], size=num_samples)

def is_imbalanced(y, threshold=0.2):
    ratios = np.bincount(y) / len(y)
    return np.min(ratios) < threshold

def generate_dummy_data(n=100):
    return pd.DataFrame({
        'Feature_1': np.random.rand(n),
        'Feature_2': np.random.rand(n),
        'Feature_3': np.random.rand(n),
        'Target': np.random.choice([0, 1, 2], size=n)
    })

# Language Map
language_map = {
    'English': 'en', 'Tamil': 'ta', 'Hindi': 'hi', 'Telugu': 'te', 'French': 'fr',
    'German': 'de', 'Spanish': 'es', 'Arabic': 'ar', 'Japanese': 'ja', 'Korean': 'ko'
}

# --- UI ---
st.title("🔠 Translation Evaluation with ML Models")

selected_model = st.sidebar.radio("Model Type", ['🎖️ Clustering', '🎖️ Regression', '🎖️ Classification'])
selected_language = st.sidebar.selectbox("Target Language", list(language_map.keys()))
lang_code = language_map[selected_language]

st.subheader("🎙️ Voice/Text Input")

# Voice input
if st.button("🎙️ Record"):
    st.session_state["input_text"] = record_audio()

# Text input
typed_text = st.text_input("Type your sentence:")

if typed_text:
    st.session_state["final_sentence"] = typed_text
elif st.session_state["input_text"]:
    st.session_state["final_sentence"] = st.session_state["input_text"]

input_text = st.session_state["final_sentence"]
st.write("✅ Final Input:", input_text)

# Centered Play button
with st.container():
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    play_clicked = st.button("▶️ Play now")
    st.markdown('</div>', unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .center-button {
        display: flex;
        justify-content: center;
        margin-top: 50px;
    }

    div.stButton > button {
        background: linear-gradient(145deg, #1f1f1f, #292929);
        border: none;
        color: #00ffff;
        font-size: 20px;
        padding: 9px 36px;
        border-radius: 60px;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0 6px 20px rgba(0, 255, 255, 0.2);
        transition: all 0.4s ease;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background: #00ffff;
        color: black;
        transform: scale(1.1) rotate(1deg);
        box-shadow: 0 0 10px #0ff, 0 0 15px #0ff, 0 0 20px #0ff;
    }

    div.stButton > button:active {
        transform: scale(0.95);
        box-shadow: inset 0 0 10px rgba(0, 200, 200, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# Translation and Audio
if input_text and play_clicked:
    translated = GoogleTranslator(source='auto', target=lang_code).translate(input_text)
    st.success(f"Translated [{selected_language}]: {translated}")
    audio_file = text_to_audio(translated, lang_code)
    if audio_file:
        st.audio(audio_file, format='audio/mp3')

# ML MODEL ZONE
X, y = generate_dummy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train_smote, y_train_smote = SMOTE().fit_resample(X_train, y_train) if is_imbalanced(y_train) else (X_train, y_train)
X_train_scaled, X_test_scaled = scale_features(X_train_smote, X_test)

if selected_model == '🎖️ Clustering':
    st.header("K-Means Clustering")
    model = KMeans(n_clusters=3).fit(X_train_scaled)
    predictions = model.predict(X_test_scaled)

elif selected_model == '🎖️ Regression':
    st.header("Linear Regression")
    model = LinearRegression().fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled).round()

elif selected_model == '🎖️ Classification':
    st.header("Classification (SVM, NB, KNN, LR)")
    models = {
        'SVM': SVC(), 'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000)
    }
    best_model, best_acc = None, 0
    results = {}
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        results[name] = acc
        if acc > best_acc:
            best_model, best_acc, best_name = clf, acc, name
    st.write(f"✅ Best Model: {best_name} ({best_acc:.2f})")
    st.bar_chart(pd.DataFrame(results.values(), index=results.keys(), columns=["Accuracy"]))
    predictions = best_model.predict(X_test_scaled)

# METRICS
st.header("📊 Evaluation Metrics")
if selected_model in ['🎖️ Clustering', '🎖️ Classification']:
    st.metric("Accuracy", f"{accuracy_score(y_test, predictions):.2f}")
    st.metric("F1-Score", f"{f1_score(y_test, predictions, average='weighted'):.2f}")
    st.dataframe(pd.DataFrame(classification_report(y_test, predictions, output_dict=True)).transpose())
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

elif selected_model == '🎖️ Regression':
    st.metric("R²", f"{r2_score(y_test, predictions):.2f}")
    st.metric("MSE", f"{mean_squared_error(y_test, predictions):.2f}")
    st.metric("MAE", f"{mean_absolute_error(y_test, predictions):.2f}")
    st.plotly_chart(px.scatter(x=y_test, y=predictions, labels={"x": "Actual", "y": "Predicted"}))

# CORRELATION
st.subheader("📌 Correlation Heatmap")
data = generate_dummy_data()
st.dataframe(data.head())
st.write("Correlation Matrix:")
st.dataframe(data.corr())
plt.figure(figsize=(10, 5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
st.pyplot(plt)
