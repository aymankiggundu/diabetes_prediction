from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

# Example usage:
# if __name__ == "__main__":
#     # Load your dataset
#     data = pd.read_csv('path_to_your_data.csv')
#     X = data.drop('target_column', axis=1)
#     y = data['target_column']
#     
#     # Split your data into training and testing sets
#     from sklearn.model_selection import train_test_split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     
#     # Initialize and train the model
#     dt_model = DecisionTreeModel()
#     dt_model.train(X_train, y_train)
#     
#     # Evaluate the model
#     accuracy = dt_model.evaluate(X_test, y_test)
#     print(f"Accuracy: {accuracy}")