import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

# =====================================
# PAGE CONFIGURATION
# =====================================
st.set_page_config(
    page_title="🎭 Emotion Detector",
    page_icon="💬",
    layout="wide",
)

# =====================================
# CUSTOM STYLES (with Navbar)
# =====================================
st.markdown("""
<style>
/* Hide default Streamlit menu and footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Remove top padding */
.block-container {
    padding-top: 0rem;
}

/* Page background */
.stApp {
    background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
    font-family: 'Poppins', sans-serif;
}

/* ==================== NAVBAR ==================== */
.navbar {
    background: linear-gradient(90deg, #667eea, #764ba2);
    padding: 1rem 2rem;
    border-radius: 0 0 20px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.navbar-title {
    font-size: 1.4rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.nav-links a {
    text-decoration: none;
    color: white;
    margin: 0 15px;
    font-weight: 500;
    transition: all 0.3s ease;
}
.nav-links a:hover {
    color: #FFD700;
    border-bottom: 2px solid #FFD700;
}

/* ==================== MAIN CARD ==================== */
.main-card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: 0px 4px 25px rgba(0,0,0,0.15);
    backdrop-filter: blur(10px);
    margin: 2rem auto;
    width: 70%;
}

/* Headings */
h1, h2, h3 {
    text-align: center;
    color: #2C3E50;
    font-weight: 600;
}

h1 span {
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-size: 1rem;
    font-weight: 500;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
}

/* Footer */
.footer {
    text-align: center;
    padding-top: 2rem;
    font-size: 0.9rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# =====================================
# LOAD MODEL
# =====================================
pipe_lr = joblib.load(open("model/text_emotion.pkl", "rb"))

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔",
    "shame": "😳", "surprise": "😮"
}

# =====================================
# FUNCTIONS
# =====================================
def predict_emotions(docx):
    return pipe_lr.predict([docx])[0]

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

# =====================================
# MAIN APP
# =====================================
def main():
    # Custom navigation bar
    st.markdown("""
    <div class="navbar">
        <div class="navbar-title">💬 Emotion Detector</div>
        <div class="nav-links">
            <a href="#home">🏠 Home</a>
            <a href="#analyze">🔍 Analyze</a>
            <a href="#about">ℹ️ About</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main content card
    st.markdown('<div class="main-card" id="home">', unsafe_allow_html=True)
    st.markdown("<h1><span>🎭 AI-Powered Emotion Detection</span></h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#34495E;'>Understand emotions hidden in your text instantly!</h4>", unsafe_allow_html=True)
    st.write("---")

    # Input form
    with st.form(key='emotion_form'):
        raw_text = st.text_area("📝 Enter your text:", height=180, placeholder="e.g., I’m feeling amazing about this new opportunity!")
        submit_button = st.form_submit_button(label='✨ Analyze Emotion')

    if submit_button:
        if not raw_text.strip():
            st.warning("⚠️ Please enter some text first.")
        else:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.subheader("🔮 Result")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.success(f"**Emotion Detected:** {prediction} {emoji_icon}")
                st.info(f"**Confidence:** {np.max(probability):.2f}")

            with col2:
                st.subheader("📊 Emotion Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion", "Probability"]

                chart = (
                    alt.Chart(proba_df_clean)
                    .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                    .encode(
                        x=alt.X('Emotion', sort='-y'),
                        y='Probability',
                        color=alt.Color('Emotion', scale=alt.Scale(scheme='turbo'))
                    )
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)

    st.markdown('<div id="about"></div>', unsafe_allow_html=True)
    st.write("---")
    st.subheader("ℹ️ About this App")
    st.write("""
    This web app uses a **machine learning model (Logistic Regression)** trained on an emotion-labeled dataset 
    to predict the emotional tone of any text you provide.  
    It leverages **Scikit-learn**, **Streamlit**, and **Altair** for an elegant, interactive experience.
    """)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="footer">💡 Built with ❤️ using Streamlit & Scikit-learn</div>', unsafe_allow_html=True)

# =====================================
# RUN APP
# =====================================
if __name__ == "__main__":
    main()







