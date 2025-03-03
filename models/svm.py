from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class SVMModel:
    def __init__(self, kernel='linear'):
        self.model = svm.SVC(kernel=kernel)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop('target', axis=1)  # Assuming 'target' is the label column
    y = data['target']
    return X, y

if __name__ == "__main__":
    # Example usage
    X, y = load_data('path_to_your_data.csv')  # Update with your data path
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_model = SVMModel(kernel='linear')
    svm_model.train(X_train, y_train)
    accuracy = svm_model.evaluate(X_test, y_test)
    print(f"SVM Model Accuracy: {accuracy}")