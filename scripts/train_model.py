import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from diabetes_prediction.models.naive_bayes import train_naive_bayes
from diabetes_prediction.models.logistic_regression import train_logistic_regression
from diabetes_prediction.models.svm import train_svm
from diabetes_prediction.models.decision_tree import train_decision_tree
from diabetes_prediction.models.knn import train_knn
from diabetes_prediction.models.xgboost import train_xgboost

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def main():
    # Load dataset
    data = load_data('diabetes_prediction/data/diabetes.csv')  # Update with actual data path
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    models = {
        'Naive Bayes': train_naive_bayes(X_train, y_train, X_test, y_test),
        'Logistic Regression': train_logistic_regression(X_train, y_train, X_test, y_test),
        'SVM': train_svm(X_train, y_train, X_test, y_test),
        'Decision Tree': train_decision_tree(X_train, y_train, X_test, y_test),
        'KNN': train_knn(X_train, y_train, X_test, y_test),
        'XGBoost': train_xgboost(X_train, y_train, X_test, y_test)
    }

    # Print accuracy of each model
    for model_name, accuracy in models.items():
        print(f'{model_name} Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    main()