import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import time
from tabulate import tabulate
from colorama import Fore, Style, init
import os
from scipy.ndimage import gaussian_filter
from sklearn.linear_model import Lasso
import numpy as np
from sklearn.preprocessing import StandardScaler

init()

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, predictions)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        sensitivity = tp / (tp + fn)
        return accuracy, precision, recall, f1, auc, sensitivity

def get_run_count(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df['Run'].nunique() + 1
    return 1

def denoise_signal(data, sigma=1):
    return gaussian_filter(data, sigma=sigma)

def sparse_learning(X_train, y_train, alpha=0.01):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train_scaled, y_train)
    return lasso, scaler

def knowledge_distillation(teacher_model, X_train, y_train, X_test, temperature=2):
    teacher_predictions = teacher_model.predict_proba(X_train) ** (1 / temperature)
    student_model = DecisionTreeClassifier()
    student_model.fit(X_train, np.argmax(teacher_predictions, axis=1))
    return student_model

def sparse_knowledge_distillation(teacher_model, X_train, y_train, X_test, alpha=0.01, temperature=2):
    lasso, scaler = sparse_learning(X_train, y_train, alpha)
    teacher_predictions = teacher_model.predict_proba(X_train) ** (1 / temperature)
    student_model = DecisionTreeClassifier()
    X_train_transformed = scaler.transform(X_train)
    student_model.fit(X_train_transformed, np.argmax(teacher_predictions, axis=1))
    return student_model

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('/Users/aymankiggundu/Desktop/BCSC/year_two/Year two Semester 2 2025/Machine Learning/ml_space_2/diabetes_prediction/data/raw/df.csv')
    X = data.drop('Diabetic', axis=1)  # Replace 'Diabetic' with the actual target column name
    y = data['Diabetic']  # Replace 'Diabetic' with the actual target column name
    
    # Split your data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    dt_model = DecisionTreeModel()
    training_time = dt_model.train(X_train, y_train)
    
    # Evaluate the model
    accuracy, precision, recall, f1, auc, sensitivity = dt_model.evaluate(X_test, y_test)
    
    # Get the current run count
    run_count = get_run_count('model_performance_metrics.csv')
    
    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'Model': ['Decision Tree'] * 6,
        'Run': [f'Run {run_count}'] * 6,
        'Metric': ['Training Time (seconds)', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'AUC'],
        'Value': [training_time, accuracy, precision, recall, f1, auc]
    })
    
    # Save the metrics to a CSV file
    try:
        existing_metrics_df = pd.read_csv('model_performance_metrics.csv')
        metrics_df.to_csv('model_performance_metrics.csv', mode='a', index=False, header=False)
    except pd.errors.EmptyDataError:
        metrics_df.to_csv('model_performance_metrics.csv', mode='w', index=False, header=True)
    
    # Create a colored version of your metrics DataFrame
    colored_metrics = []
    for index, row in metrics_df.iterrows():
        metric_name = row['Metric']
        value = row['Value']
        
        # Color code different metrics
        if 'Time' in metric_name:
            colored_metric = f"{Fore.CYAN}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.CYAN}{value:.4f}{Style.RESET_ALL}"
        elif 'Accuracy' in metric_name:
            colored_metric = f"{Fore.GREEN}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.GREEN}{value:.4f}{Style.RESET_ALL}"
        elif 'Precision' in metric_name:
            colored_metric = f"{Fore.YELLOW}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.YELLOW}{value:.4f}{Style.RESET_ALL}"
        elif 'Recall' in metric_name or 'Sensitivity' in metric_name:
            colored_metric = f"{Fore.MAGENTA}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.MAGENTA}{value:.4f}{Style.RESET_ALL}"
        elif 'F1' in metric_name:
            colored_metric = f"{Fore.BLUE}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.BLUE}{value:.4f}{Style.RESET_ALL}"
        elif 'AUC' in metric_name:
            colored_metric = f"{Fore.RED}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.RED}{value:.4f}{Style.RESET_ALL}"
        else:
            colored_metric = metric_name
            colored_value = f"{value:.4f}"
            
        colored_metrics.append([colored_metric, colored_value])
    
    # Print the colorful table
    print("\nModel Performance Metrics:")
    print(tabulate(colored_metrics, headers=['Metric', 'Value'], tablefmt='grid'))

    # Denoise the dataset
    X_denoised = denoise_signal(X.values)
    
    # Split the denoised data into training and testing sets
    X_train_denoised, X_test_denoised, y_train, y_test = train_test_split(X_denoised, y, test_size=0.2, random_state=42)
    
    # Train and evaluate the model on the denoised data
    training_time_denoised = dt_model.train(X_train_denoised, y_train)
    accuracy_denoised, precision_denoised, recall_denoised, f1_denoised, auc_denoised, sensitivity_denoised = dt_model.evaluate(X_test_denoised, y_test)
    
    # Get the current run count for denoised data
    run_count_denoised = get_run_count('model_performance_metrics.csv')
    
    # Create a DataFrame to store the metrics for denoised data
    metrics_denoised_df = pd.DataFrame({
        'Model': ['Decision Tree'] * 6,
        'Run': [f'Run {run_count_denoised} (Denoised)'] * 6,
        'Metric': ['Training Time (seconds)', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'AUC'],
        'Value': [training_time_denoised, accuracy_denoised, precision_denoised, recall_denoised, f1_denoised, auc_denoised]
    })
    
    # Save the metrics for denoised data to a CSV file
    try:
        existing_metrics_df = pd.read_csv('model_performance_metrics.csv')
        metrics_denoised_df.to_csv('model_performance_metrics.csv', mode='a', index=False, header=False)
    except pd.errors.EmptyDataError:
        metrics_denoised_df.to_csv('model_performance_metrics.csv', mode='w', index=False, header=True)
    
    # Create a colored version of your metrics DataFrame for denoised data
    colored_metrics_denoised = []
    for index, row in metrics_denoised_df.iterrows():
        metric_name = row['Metric']
        value = row['Value']
        
        # Color code different metrics
        if 'Time' in metric_name:
            colored_metric = f"{Fore.CYAN}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.CYAN}{value:.4f}{Style.RESET_ALL}"
        elif 'Accuracy' in metric_name:
            colored_metric = f"{Fore.GREEN}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.GREEN}{value:.4f}{Style.RESET_ALL}"
        elif 'Precision' in metric_name:
            colored_metric = f"{Fore.YELLOW}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.YELLOW}{value:.4f}{Style.RESET_ALL}"
        elif 'Recall' in metric_name or 'Sensitivity' in metric_name:
            colored_metric = f"{Fore.MAGENTA}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.MAGENTA}{value:.4f}{Style.RESET_ALL}"
        elif 'F1' in metric_name:
            colored_metric = f"{Fore.BLUE}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.BLUE}{value:.4f}{Style.RESET_ALL}"
        elif 'AUC' in metric_name:
            colored_metric = f"{Fore.RED}{metric_name}{Style.RESET_ALL}"
            colored_value = f"{Fore.RED}{value:.4f}{Style.RESET_ALL}"
        else:
            colored_metric = metric_name
            colored_value = f"{value:.4f}"
            
        colored_metrics_denoised.append([colored_metric, colored_value])
    
    # Print the colorful table for denoised data
    print("\nModel Performance Metrics (Denoised):")
    print(tabulate(colored_metrics_denoised, headers=['Metric', 'Value'], tablefmt='grid'))

    # Sparse learning on original data
    lasso_model, scaler = sparse_learning(X_train, y_train)
    sparse_training_time = time.time() - training_time
    sparse_predictions = lasso_model.predict(scaler.transform(X_test))
    sparse_predictions_binary = (sparse_predictions > 0.5).astype(int)
    sparse_accuracy = accuracy_score(y_test, sparse_predictions_binary)
    sparse_precision = precision_score(y_test, sparse_predictions_binary)
    sparse_recall = recall_score(y_test, sparse_predictions_binary)
    sparse_f1 = f1_score(y_test, sparse_predictions_binary)
    sparse_auc = roc_auc_score(y_test, sparse_predictions_binary)
    
    # Knowledge distillation on original data
    kd_start_time = time.time()
    student_model = knowledge_distillation(dt_model.model, X_train, y_train, X_test)
    kd_training_time = time.time() - kd_start_time
    kd_predictions = student_model.predict(X_test)
    kd_predictions_binary = (kd_predictions > 0.5).astype(int)
    kd_accuracy = accuracy_score(y_test, kd_predictions_binary)
    kd_precision = precision_score(y_test, kd_predictions_binary)
    kd_recall = recall_score(y_test, kd_predictions_binary)
    kd_f1 = f1_score(y_test, kd_predictions_binary)
    kd_auc = roc_auc_score(y_test, kd_predictions_binary)
    
    # Sparse knowledge distillation on original data
    sparse_kd_start_time = time.time()
    sparse_kd_model = sparse_knowledge_distillation(dt_model.model, X_train, y_train, X_test)
    sparse_kd_training_time = time.time() - sparse_kd_start_time
    sparse_kd_predictions = sparse_kd_model.predict(scaler.transform(X_test))
    sparse_kd_predictions_binary = (sparse_kd_predictions > 0.5).astype(int)
    sparse_kd_accuracy = accuracy_score(y_test, sparse_kd_predictions_binary)
    sparse_kd_precision = precision_score(y_test, sparse_kd_predictions_binary)
    sparse_kd_recall = recall_score(y_test, sparse_kd_predictions_binary)
    sparse_kd_f1 = f1_score(y_test, sparse_kd_predictions_binary)
    sparse_kd_auc = roc_auc_score(y_test, sparse_kd_predictions_binary)
    
    # Sparse learning on denoised data
    lasso_model_denoised, scaler_denoised = sparse_learning(X_train_denoised, y_train)
    sparse_training_time_denoised = time.time() - training_time_denoised
    sparse_predictions_denoised = lasso_model_denoised.predict(scaler_denoised.transform(X_test_denoised))
    sparse_predictions_denoised_binary = (sparse_predictions_denoised > 0.5).astype(int)
    sparse_accuracy_denoised = accuracy_score(y_test, sparse_predictions_denoised_binary)
    sparse_precision_denoised = precision_score(y_test, sparse_predictions_denoised_binary)
    sparse_recall_denoised = recall_score(y_test, sparse_predictions_denoised_binary)
    sparse_f1_denoised = f1_score(y_test, sparse_predictions_denoised_binary)
    sparse_auc_denoised = roc_auc_score(y_test, sparse_predictions_denoised_binary)
    
    # Knowledge distillation on denoised data
    kd_start_time_denoised = time.time()
    student_model_denoised = knowledge_distillation(dt_model.model, X_train_denoised, y_train, X_test_denoised)
    kd_training_time_denoised = time.time() - kd_start_time_denoised
    kd_predictions_denoised = student_model_denoised.predict(X_test_denoised)
    kd_predictions_denoised_binary = (kd_predictions_denoised > 0.5).astype(int)
    kd_accuracy_denoised = accuracy_score(y_test, kd_predictions_denoised_binary)
    kd_precision_denoised = precision_score(y_test, kd_predictions_denoised_binary)
    kd_recall_denoised = recall_score(y_test, kd_predictions_denoised_binary)
    kd_f1_denoised = f1_score(y_test, kd_predictions_denoised_binary)
    kd_auc_denoised = roc_auc_score(y_test, kd_predictions_denoised_binary)
    
    # Sparse knowledge distillation on denoised data
    sparse_kd_start_time_denoised = time.time()
    sparse_kd_model_denoised = sparse_knowledge_distillation(dt_model.model, X_train_denoised, y_train, X_test_denoised)
    sparse_kd_training_time_denoised = time.time() - sparse_kd_start_time_denoised
    sparse_kd_predictions_denoised = sparse_kd_model_denoised.predict(scaler_denoised.transform(X_test_denoised))
    sparse_kd_predictions_denoised_binary = (sparse_kd_predictions_denoised > 0.5).astype(int)
    sparse_kd_accuracy_denoised = accuracy_score(y_test, sparse_kd_predictions_denoised_binary)
    sparse_kd_precision_denoised = precision_score(y_test, sparse_kd_predictions_denoised_binary)
    sparse_kd_recall_denoised = recall_score(y_test, sparse_kd_predictions_denoised_binary)
    sparse_kd_f1_denoised = f1_score(y_test, sparse_kd_predictions_denoised_binary)
    sparse_kd_auc_denoised = roc_auc_score(y_test, sparse_kd_predictions_denoised_binary)
    
    # Create DataFrames to store the metrics for additional scenarios
    additional_metrics_df = pd.DataFrame({
        'Model': ['Sparse Learning', 'Knowledge Distillation', 'Sparse Knowledge Distillation'] * 6,
        'Run': [f'Run {run_count}'] * 18,
        'Metric': ['Training Time (seconds)', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'AUC'] * 3,
        'Value': [
            sparse_training_time, sparse_accuracy, sparse_precision, sparse_recall, sparse_f1, sparse_auc,
            kd_training_time, kd_accuracy, kd_precision, kd_recall, kd_f1, kd_auc,
            sparse_kd_training_time, sparse_kd_accuracy, sparse_kd_precision, sparse_kd_recall, sparse_kd_f1, sparse_kd_auc
        ]
    })
    
    additional_metrics_denoised_df = pd.DataFrame({
        'Model': ['Sparse Learning (Denoised)', 'Knowledge Distillation (Denoised)', 'Sparse Knowledge Distillation (Denoised)'] * 6,
        'Run': [f'Run {run_count_denoised} (Denoised)'] * 18,
        'Metric': ['Training Time (seconds)', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'AUC'] * 3,
        'Value': [
            sparse_training_time_denoised, sparse_accuracy_denoised, sparse_precision_denoised, sparse_recall_denoised, sparse_f1_denoised, sparse_auc_denoised,
            kd_training_time_denoised, kd_accuracy_denoised, kd_precision_denoised, kd_recall_denoised, kd_f1_denoised, kd_auc_denoised,
            sparse_kd_training_time_denoised, sparse_kd_accuracy_denoised, sparse_kd_precision_denoised, sparse_kd_recall_denoised, sparse_kd_f1_denoised, sparse_kd_auc_denoised
        ]
    })
    
    # Save the additional metrics to the CSV file
    try:
        existing_metrics_df = pd.read_csv('model_performance_metrics.csv')
        additional_metrics_df.to_csv('model_performance_metrics.csv', mode='a', index=False, header=False)
        additional_metrics_denoised_df.to_csv('model_performance_metrics.csv', mode='a', index=False, header=False)
    except pd.errors.EmptyDataError:
        additional_metrics_df.to_csv('model_performance_metrics.csv', mode='w', index=False, header=True)
        additional_metrics_denoised_df.to_csv('model_performance_metrics.csv', mode='w', index=False, header=True)
    
    # Print results for sparse learning, knowledge distillation, and sparse knowledge distillation
    print("\nSparse Learning Metrics (Original Data):")
    print(f"Accuracy: {sparse_accuracy:.4f}, Precision: {sparse_precision:.4f}, Recall: {sparse_recall:.4f}, F1 Score: {sparse_f1:.4f}, AUC: {sparse_auc:.4f}")
    
    print("\nKnowledge Distillation Metrics (Original Data):")
    print(f"Accuracy: {kd_accuracy:.4f}, Precision: {kd_precision:.4f}, Recall: {kd_recall:.4f}, F1 Score: {kd_f1:.4f}, AUC: {kd_auc:.4f}")
    
    print("\nSparse Knowledge Distillation Metrics (Original Data):")
    print(f"Accuracy: {sparse_kd_accuracy:.4f}, Precision: {sparse_kd_precision:.4f}, Recall: {sparse_kd_recall:.4f}, F1 Score: {sparse_kd_f1:.4f}, AUC: {sparse_kd_auc:.4f}")
    
    print("\nSparse Learning Metrics (Denoised Data):")
    print(f"Accuracy: {sparse_accuracy_denoised:.4f}, Precision: {sparse_precision_denoised:.4f}, Recall: {sparse_recall_denoised:.4f}, F1 Score: {sparse_f1_denoised:.4f}, AUC: {sparse_auc_denoised:.4f}")
    
    print("\nKnowledge Distillation Metrics (Denoised Data):")
    print(f"Accuracy: {kd_accuracy_denoised:.4f}, Precision: {kd_precision_denoised:.4f}, Recall: {kd_recall_denoised:.4f}, F1 Score: {kd_f1_denoised:.4f}, AUC: {kd_auc_denoised:.4f}")
    
    print("\nSparse Knowledge Distillation Metrics (Denoised Data):")
    print(f"Accuracy: {sparse_kd_accuracy_denoised:.4f}, Precision: {sparse_kd_precision_denoised:.4f}, Recall: {sparse_kd_recall_denoised:.4f}, F1 Score: {sparse_kd_f1_denoised:.4f}, AUC: {sparse_kd_auc_denoised:.4f}")

# Save the trained model to a .pkl file
joblib.dump(dt_model.model, 'decision_tree_model.pkl')