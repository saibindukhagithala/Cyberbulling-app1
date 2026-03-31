import streamlit as st
import joblib
import re
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load('cyberbullying_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# -----------------------------
# Offensive words list
# -----------------------------
offensive_words = ["stupid", "idiot", "hate", "loser", "dumb", "kill", "ugly", "shut up"]

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Cyberbullying Detection 🚀",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("About")
st.sidebar.info("""
This app detects cyberbullying messages in live chat.
- Offensive words are highlighted in red.
- Safe messages show a green bubble.
- Alert sound on offensive messages.
- Pop sound on safe messages.
- Download chat history or view word cloud.
- Developed by Batch 6
""")

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Cyberbullying Detection App 🛡️</h1>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Chat history
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Input
# -----------------------------
user_input = st.text_area("💬 Type your message here...", height=120)

# -----------------------------
# Highlight offensive words
# -----------------------------
def highlight_offensive(message):
    highlighted = message
    for word in offensive_words:
        pattern = re.compile(rf"({word})", re.IGNORECASE)
        highlighted = pattern.sub(r"<span style='color:red; font-weight:bold;'>\1</span>", highlighted)
    return highlighted

# -----------------------------
# Play WAV sounds
# -----------------------------
def play_sound(file_path):
    audio_file = open(file_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')  # works for WAV files

# -----------------------------
# Detect message
# -----------------------------
def detect_message(text):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text to check")
        return

    # Vectorize and predict
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    # Sentiment analysis
    sentiment_score = TextBlob(text).sentiment.polarity
    if sentiment_score > 0.2:
        sentiment = "😊 Positive"
    elif sentiment_score < -0.2:
        sentiment = "😡 Negative"
    else:
        sentiment = "😐 Neutral"

    # Severity
    words = text.split()
    offense_count = sum(1 for w in words if w.lower() in offensive_words)
    severity = round((offense_count / len(words)) * 100, 2) if words else 0

    # Update history
    if prediction == 1:
        st.session_state.history.append(("🚨 Cyberbullying Detected!", text, "danger", sentiment, severity))
        play_sound("alert_sound.wav")
    else:
        st.session_state.history.append(("😊 Safe Message!", text, "safe", sentiment, severity))
        play_sound("pop_sound.wav")

# -----------------------------
# Detect button
# -----------------------------
if st.button("🔍 Detect Cyberbullying"):
    detect_message(user_input)

# -----------------------------
# Display chat history
# -----------------------------
for label, message, status, sentiment, severity in reversed(st.session_state.history):
    highlighted_message = highlight_offensive(message)
    if status == "danger":
        st.markdown(f"""
        <div class="flicker-danger" style='padding:12px; border-radius:10px; margin-bottom:5px;'>
            <b>{label}</b> | {sentiment} | Severity: {severity}%<br>{highlighted_message}
        </div>
        <style>
        @keyframes flickerRed {{
            0% {{ background-color:#FFCCCC; }}
            50% {{ background-color:#FF6666; }}
            100% {{ background-color:#FFCCCC; }}
        }}
        .flicker-danger {{
            animation: flickerRed 0.5s ease-in-out;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="flicker-safe" style='padding:12px; border-radius:10px; margin-bottom:5px;'>
            <b>{label}</b> | {sentiment} | Severity: {severity}%<br>{highlighted_message}
        </div>
        <style>
        @keyframes flickerGreen {{
            0% {{ background-color:#CCFFCC; }}
            50% {{ background-color:#66FF66; }}
            100% {{ background-color:#CCFFCC; }}
        }}
        .flicker-safe {{
            animation: flickerGreen 0.5s ease-in-out;
        }}
        </style>
        """, unsafe_allow_html=True)

# -----------------------------
# Word Cloud of offensive words
# -----------------------------
if st.sidebar.button("Show Word Cloud"):
    text = " ".join([msg for _, msg, status, _, _ in st.session_state.history if status == "danger"])
    if text:
        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt.gcf())
    else:
        st.info("No offensive words detected yet.")

# -----------------------------
# Download chat history
# -----------------------------
if st.sidebar.button("Download Chat History"):
    df = pd.DataFrame(st.session_state.history, columns=["Label", "Message", "Status", "Sentiment", "Severity"])
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, file_name="chat_history.csv")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made with ❤️ for safer online chatting</p>", unsafe_allow_html=True)
