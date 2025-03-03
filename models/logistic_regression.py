from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class LogisticRegressionModel:
    def __init__(self):
        self.model = LogisticRegression()

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        X = data.drop('target', axis=1)  # Assuming 'target' is the label column
        y = data['target']
        return X, y

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filename):
        import joblib
        joblib.dump(self.model, filename)

    def load_model(self, filename):
        import joblib
        self.model = joblib.load(filename)