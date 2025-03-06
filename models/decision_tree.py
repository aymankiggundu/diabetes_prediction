from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import time
from tabulate import tabulate
from colorama import Fore, Style, init

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

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    data = pd.read_csv('./data/raw/df.csv')
    X = data.drop('Diabetic', axis=1)
    y = data['Diabetic']
    
    # Split your data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    dt_model = DecisionTreeModel()
    training_time = dt_model.train(X_train, y_train)
    
    # Evaluate the model
    accuracy, precision, recall, f1, auc, sensitivity = dt_model.evaluate(X_test, y_test)
    
    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Training Time (seconds)', 'Accuracy', 'Precision', 'Recall (Sensitivity)', 'F1 Score', 'AUC'],
        'Value': [training_time, accuracy, precision, recall, f1, auc]
    })
    
    # Save the metrics to a CSV file
    metrics_df.to_csv('model_performance_metrics.csv', index=False)
    
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
