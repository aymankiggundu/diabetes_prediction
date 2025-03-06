import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split

# specify parameters
fs = 2  # Hz, sampling frequency
N = 4096  # number of points for FFT analysis

# memory allocation for Dominant Frequency (DF) calculation
df = np.zeros((40, 6))  # Updated to 40 entries

# bandpass filter for noise reduction
b, a = butter(3, [0.03, 0.25], btype='bandpass', fs=fs)

# DF calculation for existing 20 participants
for ind in range(1, 21):
    # FASTING -------------------------------------------------------------
    
    # signal loading
    file_name_f = f'ID{ind}_fasting.txt'
    dat_f = np.loadtxt(file_name_f)
    
    # EGG filtering
    ch1_f = filtfilt(b, a, dat_f[:2400, 0])  # channel 1
    ch2_f = filtfilt(b, a, dat_f[:2400, 1])  # channel 2
    ch3_f = filtfilt(b, a, dat_f[:2400, 2])  # channel 3
    
    # FFT
    fftch1_f = np.abs(fft(ch1_f, N)[:N//2+1])**2
    fftch2_f = np.abs(fft(ch2_f, N)[:N//2+1])**2
    fftch3_f = np.abs(fft(ch3_f, N)[:N//2+1])**2
    
    # automatic DF calculation
    df[ind-1, 0] = np.argmax(fftch1_f) / 2048
    df[ind-1, 1] = np.argmax(fftch2_f) / 2048
    df[ind-1, 2] = np.argmax(fftch3_f) / 2048
    
    # POSTPRANDIAL --------------------------------------------------------
    
    # signal loading
    file_name_p = f'ID{ind}_postprandial.txt'
    dat_p = np.loadtxt(file_name_p)

    # EGG filtering
    ch1_p = filtfilt(b, a, dat_p[:2400, 0])  # channel 1
    ch2_p = filtfilt(b, a, dat_p[:2400, 1])  # channel 2
    ch3_p = filtfilt(b, a, dat_p[:2400, 2])  # channel 3
    
    # FFT
    fftch1_p = np.abs(fft(ch1_p, N)[:N//2+1])**2
    fftch2_p = np.abs(fft(ch2_p, N)[:N//2+1])**2
    fftch3_p = np.abs(fft(ch3_p, N)[:N//2+1])**2
    
    # automatic DF calculation
    df[ind-1, 3] = np.argmax(fftch1_p) / 2048
    df[ind-1, 4] = np.argmax(fftch2_p) / 2048
    df[ind-1, 5] = np.argmax(fftch3_p) / 2048

# conversion from Hz to cpm (cycles per minute)
df[:20] *= 60

# manual corrections for false DF values
df[3, 3] = 3.1934  # ID4 postprandial channel 1
df[3, 4] = 3.1348  # ID4 postprandial channel 2
df[3, 5] = 3.1348  # ID4 postprandial channel 3
df[5, 2] = 2.4900  # ID6 fasting channel 3
df[14, 3] = 2.2560  # ID15 postprandial channel 1
df[16, 0] = 2.9592  # ID17 fasting channel 1
df[16, 1] = 3.0469  # ID17 fasting channel 2
df[16, 2] = 3.0762  # ID17 fasting channel 3

# Generate synthetic data for 20 more participants (15 diabetic, 5 non-diabetic)
np.random.seed(42)  # For reproducibility

# Generate synthetic data for 15 diabetic participants
for ind in range(21, 36):
    df[ind-1, :3] = np.random.uniform(2.0, 3.0, 3)  # Fasting
    df[ind-1, 3:] = np.random.uniform(3.0, 4.0, 3)  # Postprandial

# Generate synthetic data for 5 non-diabetic participants
for ind in range(36, 41):
    df[ind-1, :3] = np.random.uniform(2.5, 3.5, 3)  # Fasting
    df[ind-1, 3:] = np.random.uniform(2.5, 3.5, 3)  # Postprandial

# Add labels for diabetic (1) and non-diabetic (0)
labels = np.zeros(40)
labels[20:35] = 1  # Mark the 15 diabetic participants

# Combine df and labels into a single DataFrame
df_df = pd.DataFrame(np.hstack((df, labels.reshape(-1, 1))), columns=[
    'Fasting_Ch1', 'Fasting_Ch2', 'Fasting_Ch3', 'Postprandial_Ch1', 'Postprandial_Ch2', 'Postprandial_Ch3', 'Diabetic'])

# Shuffle and split the dataset into training, validation, and testing sets
train_df, temp_df = train_test_split(df_df, test_size=0.3, random_state=42, stratify=df_df['Diabetic'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['Diabetic'])

# Save the splits to CSV files
train_df.to_csv('train_df.csv', index=False)
val_df.to_csv('val_df.csv', index=False)
test_df.to_csv('test_df.csv', index=False)

# paired-sample t-tests

# for all subjects ID1-ID20
p_values = []
h_values = []
for i in range(3):
    h, p = ttest_rel(df[:20, i], df[:20, i+3])
    p_values.append(p)
    h_values.append(h)

# for subjects with lower body mass index (BMI)
df_low = np.vstack([df[:3, :], df[8, :], df[11, :], df[13, :], df[15:18, :]])
for i in range(3):
    h, p = ttest_rel(df_low[:, i], df_low[:, i+3])
    p_values.append(p)
    h_values.append(h)

# for subjects ID11-ID20 (higher BMI)
df_high = np.vstack([df[3:8, :], df[9:10, :], df[12, :], df[14, :], df[19, :]])
for i in range(3):
    h, p = ttest_rel(df_high[:, i], df_high[:, i+3])
    p_values.append(p)
    h_values.append(h)

# show p-values and test decisions
print("p-values:", p_values)
print("h-values:", h_values)

