import pandas as pd
import numpy as np
import os

# Load the CSV file
df = pd.read_csv('/Users/aymankiggundu/Desktop/BCSC/year_two/Year two Semester 2 2025/Machine Learning/ml_space_2/diabetes_prediction/data/raw/df.csv')

# Define the output directory
output_dir = '/Users/aymankiggundu/Desktop/BCSC/year_two/Year two Semester 2 2025/Machine Learning/ml_space_2/diabetes_prediction/data/raw/'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to generate synthetic data
def generate_synthetic_data(num_samples=50):
    return np.random.randn(num_samples, 3)

# Generate synthetic data files for the 15 diabetic patients
start_id = 21
num_patients = 15
for i in range(num_patients):
    individual_id = start_id + i
    fasting_data = generate_synthetic_data()
    postprandial_data = generate_synthetic_data()
    
    # Save fasting data
    fasting_file_path = os.path.join(output_dir, f'ID{individual_id}_fasting.txt')
    np.savetxt(fasting_file_path, fasting_data, fmt='%.3f')
    
    # Save postprandial data
    postprandial_file_path = os.path.join(output_dir, f'ID{individual_id}_postprandial.txt')
    np.savetxt(postprandial_file_path, postprandial_data, fmt='%.3f')

print("Synthetic data files for 15 diabetic patients generated successfully.")
