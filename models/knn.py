from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNNModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        return accuracy_score(y_test, predictions)

def main():
    # Load dataset
    data = pd.read_csv('path_to_your_dataset.csv')  # Update with the actual path
    X = data.drop('target_column', axis=1)  # Update with the actual target column name
    y = data['target_column']  # Update with the actual target column name

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the KNN model
    knn_model = KNNModel(n_neighbors=5)
    knn_model.train(X_train, y_train)

    # Evaluate the model
    accuracy = knn_model.evaluate(X_test, y_test)
    print(f'KNN Model Accuracy: {accuracy:.2f}')

if __name__ == "__main__":
    main()