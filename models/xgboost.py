from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class XGBoostModel:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        report = classification_report(y, predictions)
        return accuracy, report

def main():
    # Load dataset
    data = pd.read_csv('path_to_your_dataset.csv')  # Update with your dataset path
    X = data.drop('target_column', axis=1)  # Update with your feature columns
    y = data['target_column']  # Update with your target column

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)

    # Evaluate the model
    accuracy, report = xgb_model.evaluate(X_test, y_test)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    main()