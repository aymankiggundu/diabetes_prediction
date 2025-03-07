from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from scipy.signal import butter, filtfilt

app = Flask(__name__)

# Load the trained decision tree model
model = joblib.load('/Users/aymankiggundu/Desktop/BCSC/year_two/Year two Semester 2 2025/Machine Learning/ml_space_2/diabetes_prediction/models/decision_tree_model.pkl')

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    fasting_file = request.files['fastingFile']
    postprandial_file = request.files['postprandialFile']
    
    fasting_df = preprocess_signal(fasting_file)
    postprandial_df = preprocess_signal(postprandial_file)
    
    # Combine the data for prediction
    combined_df = np.array(fasting_df + postprandial_df).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(combined_df)
    probability = model.predict_proba(combined_df)
    
    result = 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'
    confidence = probability[0][prediction[0]] * 100
    
    return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    app.run(port=5000)
