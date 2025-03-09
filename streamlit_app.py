import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.signal import butter, filtfilt

# Load the trained decision tree model
model = joblib.load('/Users/aymankiggundu/Desktop/BCSC/year_two/Year two Semester 2 2025/Machine Learning/ml_space_2/diabetes_prediction/models/decision_tree_model.pkl')

st.set_page_config(page_title="Diabetes Prediction by EGG Analysis", layout="wide")

st.title('Diabetes Prediction by EGG Analysis')
st.write('This web application predicts diabetes based on Electrogastrogram (EGG) signals.')

st.sidebar.header('Upload Data Files')
st.sidebar.write("Please upload your files in the format of 'ID1_fasting.txt' and 'ID1_postprandial.txt'.")
uploaded_fasting_file = st.sidebar.file_uploader('Fasting State TXT', type='txt')
uploaded_postprandial_file = st.sidebar.file_uploader('Postprandial State TXT', type='txt')

def preprocess_signal(file):
    data = np.loadtxt(file)
    fs = 2  # Hz, sampling frequency
    N = 4096  # number of points for FFT analysis
    b, a = butter(3, [0.03, 0.25], btype='bandpass', fs=fs)
    
    # EGG filtering
    ch1 = filtfilt(b, a, data[:, 0])
    ch2 = filtfilt(b, a, data[:, 1])
    ch3 = filtfilt(b, a, data[:, 2])
    
    # FFT
    fftch1 = np.abs(np.fft.fft(ch1, N))**2
    fftch2 = np.abs(np.fft.fft(ch2, N))**2
    fftch3 = np.abs(np.fft.fft(ch3, N))**2
    
    # Dominant frequency calculation
    df1 = np.argmax(fftch1[:N//2+1]) / (N / fs)
    df2 = np.argmax(fftch2[:N//2+1]) / (N / fs)
    df3 = np.argmax(fftch3[:N//2+1]) / (N / fs)
    
    return [df1 * 60, df2 * 60, df3 * 60]  # Convert to cycles per minute

@st.cache_data
def predict(fasting_file, postprandial_file):
    fasting_df = preprocess_signal(fasting_file)
    postprandial_df = preprocess_signal(postprandial_file)
    
    # Combine the data for prediction
    combined_df = np.array(fasting_df + postprandial_df).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(combined_df)
    probability = model.predict_proba(combined_df)
    
    result = 'Diabetic' if prediction[0] == 1 else 'Healthy'
    confidence = probability[0][1] * 100 if prediction[0] == 1 else probability[0][0] * 100
    
    return result, confidence

if st.sidebar.button("Predict"):
    if uploaded_fasting_file and uploaded_postprandial_file:
        result, confidence = predict(uploaded_fasting_file, uploaded_postprandial_file)
        st.subheader('Prediction Result')
        st.markdown(f"""
        <div class="alert {'alert-warning' if result == 'Diabetic' else 'alert-success'}" style="font-family: 'Garamond', cursive; font-size: 24px; font-weight: bold;">
            <strong>Prediction:</strong> {result}<br>
            <strong>Confidence:</strong> {confidence:.2f}%
        </div>
        """, unsafe_allow_html=True)
        button_color = "red" if result == "Diabetic" else "green"
        st.markdown(f"""
        <style>
        .stButton>button {{
            background-color: {button_color};
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 12px;
            transition-duration: 0.4s;
        }}
        .stButton>button:hover {{
            background-color: white;
            color: {button_color};
            border: 2px solid {button_color};
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.error("Please upload both TXT files.")

st.markdown("""
<div class="card mt-4">
    <div class="card-body">
        <h5 class="card-title">Instructions</h5>
        <ol>
            <li>
                Upload two TXT files:
                <ul>
                    <li>
                        One for fasting state (3 columns for channels 1, 2, and 3)
                    </li>
                    <li>
                        One for postprandial state (3 columns for channels 1, 2, and 3)
                    </li>
                </ul>
            </li>
            <li>
                Each file should have exactly 3 columns representing the three channels
            </li>
            <li>The data should be numeric values</li>
            <li>Click 'Predict' to get the prediction result</li>
        </ol>
        <div class="alert alert-info">
            <strong>Example TXT Format:</strong><br />
            <code>
                0.5,0.6,0.7<br />
                0.8,0.9,1.0<br />
                ...
            </code>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
