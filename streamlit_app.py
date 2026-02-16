import streamlit as st
from dataset_collector import collect_for_class
from train_lstm import train
from real_time_recognition import run_realtime
from utils import speech_to_text, speak_text
import threading, os

st.set_page_config(page_title='Sign Language Detector', layout='wide')

# ------------ Light Professional Dashboard UI Styling (NO logic change) ------------
st.markdown("""
<style>
/* Base/Global Styles */
body {
    background: #f0f2f6; /* Very light grey/blue for professional feel */
    font-family: 'Inter', sans-serif;
}

/* Main Title - Animated */
.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
    color: #1a4f8d; /* Professional deep blue */
    padding: 25px 0;
    margin-bottom: 30px;
    border-bottom: 3px solid #e0e4eb;
    animation: slideInDown 1s ease-out;
}

@keyframes slideInDown {
    from { opacity: 0; transform: translateY(-30px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Card Container - Animated */
.card {
    background: white;
    padding: 35px;
    border-radius: 15px;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08); /* Clean, subtle shadow */
    margin-bottom: 30px;
    border-left: 6px solid #4a90e2; /* Modern accent color */
    transition: transform 0.3s ease-in-out;
    animation: fadeIn 1.2s ease-out;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.12);
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Streamlit Header (h2) */
h2 {
    color: #2c3e50; /* Darker text for readability */
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 20px;
}

/* Primary Button - Animated Glow */
.stButton>button {
    background: linear-gradient(45deg, #4a90e2, #2980b9) !important; /* Blue gradient */
    color: white !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    height: 3.5em !important;
    padding: 0 25px !important;
    font-weight: bold;
    border: none !important;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(74, 144, 226, 0.4);
}

.stButton>button:hover {
    background: linear-gradient(45deg, #2980b9, #4a90e2) !important;
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.6);
}

/* Info/Status Box - Clean & Subtle */
.info-box {
    background: #eef7ff; /* Very light blue */
    border-left: 5px solid #4a90e2;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    margin-bottom: 20px;
    font-weight: 500;
    color: #34495e;
    animation: slideInRight 0.6s ease-out;
}

@keyframes slideInRight {
    from { transform: translateX(20px); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    justify-content: space-around;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #8c98a6; /* Greyed out tabs */
    background-color: transparent;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 25px;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    color: #1a4f8d !important; /* Selected tab is deep blue */
    background-color: white;
    border-top: 4px solid #4a90e2 !important;
    border-bottom: none !important;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<div class="main-title">🌐 Medical Sign Language Detector – Doctor & Patient Communication System</div>', unsafe_allow_html=True)

# Define Tabs
tabs = st.tabs(['📸 Data Collection','🧠 Model Training','🧑‍⚕️ Patient Dashboard','👨‍⚕️ Doctor Dashboard'])

# -------- TAB 1: Data Collection --------
with tabs[0]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('📸 Data Acquisition Module')
    st.markdown('<div class="info-box">**Purpose:** Capture and label keypoint data for new signs using the integrated webcam.</div>', unsafe_allow_html=True)

    class_name = st.text_input('**Sign Label:** (e.g., headache, urgent, thank you)')
    samples = st.number_input('**Required Samples:**', 10, 2000, 300)

    if st.button('🎥 Start Data Collection', key='collect_btn', type='primary'):
        if not class_name.strip():
            st.error('❌ Please define a label for the sign before starting.')
        else:
            with st.spinner(f'Initiating webcam capture for **{class_name.strip()}**...'):
                # LOGIC: Calls the data collection function
                collect_for_class(class_name.strip(), samples)
            st.success(f'✅ **Success:** {samples} keypoint sequences collected for "{class_name.strip()}".')

    st.markdown('</div>', unsafe_allow_html=True)

# -------- TAB 2: Model Training --------
with tabs[1]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('🧠 LSTM Model Management')
    st.markdown('<div class="info-box">**Workflow:** Preprocess raw keypoints into sequences, then train the model to recognize patterns.</div>', unsafe_allow_html=True)

    # Preprocessing Section
    st.subheader('1. Data Preparation')
    if st.button('⚙️ Run Data Preprocessing', key='prep_btn'):
        from preprocess import build_sequences_from_keypoints
        with st.spinner('Preparing sequences and building dataset.npz...'):
            try:
                # LOGIC: Calls the preprocessing function
                build_sequences_from_keypoints()
                st.success('✅ **Preprocessing Complete:** Dataset built and ready for training.')
            except Exception as e:
                st.error(f'❌ **Preprocessing Error:** Check raw data folders. Error: {e}')

    st.subheader('2. Training Parameters')
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input('**Epochs** (Number of training rounds)', 1, 500, 80)
    with col2:
        batch = st.number_input('**Batch Size**', 8, 512, 32)

    # Training Section
    if st.button('🚀 Initiate Model Training', key='train_btn', type='primary'):
        with st.spinner('⏳ Training LSTM model (may take several minutes)...'):
            # LOGIC: Calls the training function
            train('data/sequences/dataset.npz', int(epochs), int(batch))
            st.success('✅ **Training Complete:** New model saved and deployed for recognition.')

    st.markdown('</div>', unsafe_allow_html=True)

# -------- TAB 3: Patient Dashboard --------
with tabs[2]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('🧑‍⚕️ Sign Recognition Interface')
    
    st.markdown('<div class="info-box">**Mode:** Real-time sign detection using the pre-trained model. A separate, high-performance OpenCV window will launch.</div>', unsafe_allow_html=True)

    if st.button('▶️ Activate Real-Time Detection', key='realtime_btn', type='primary'):
        st.info('Launching external recognition window... Please keep this Streamlit tab open.')
        # LOGIC: Starts the real-time recognition in a separate thread
        threading.Thread(target=run_realtime, daemon=True).start()

    st.markdown('</div>', unsafe_allow_html=True)

# -------- TAB 4: Doctor Dashboard --------
with tabs[3]:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header('👨‍⚕️ Two-Way Speech & Text Translator')
    
    st.markdown('<div class="info-box">**Function:** Facilitate communication between users who rely on speech and those who need text/sign output.</div>', unsafe_allow_html=True)

    # Text-to-Speech Section
    st.subheader('Text-to-Speech (System Voice)')
    speak = st.text_area('💬 **Input Text to be Spoken Aloud:**', height=100)

    if st.button('🔊 Synthesize and Speak', key='speak_btn', type='primary'):
        if speak.strip():
            # LOGIC: Starts text-to-speech in a separate thread
            threading.Thread(target=speak_text, args=(speak,), daemon=True).start()
            st.info('Voice output initiated.')
        else:
            st.warning('⚠️ Text box is empty. Enter text to generate speech.')

    # Speech-to-Text Section
    st.subheader('Speech-to-Text (Microphone Input)')
    if st.button('🎙️ Capture and Transcribe Speech', key='transcribe_btn'):
        st.info('Actively listening... Speak clearly into the microphone.')
        # LOGIC: Calls the speech-to-text function
        txt = speech_to_text()
        if txt:
            st.success(f'📝 **Transcription Result:** {txt}')
        else:
            st.error('❌ Failed to capture or transcribe speech. Please ensure the microphone is working.')

    st.markdown('</div>', unsafe_allow_html=True)