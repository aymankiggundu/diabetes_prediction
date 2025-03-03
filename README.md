# Machine Learning Diabetes Prediction Project

This project, `ml_space_2`, is focused on predicting diabetes using various machine learning models. The goal is to implement and compare different algorithms to determine their effectiveness in predicting diabetes based on a given dataset.

## Project Structure

The project is organized into the following directories and files:

- `diabetes_prediction/`: Contains all files related to the diabetes prediction models.
  - `data/`: Directory for dataset-related files.
  - `models/`: Contains implementations of various machine learning models.
  - `notebooks/`: Jupyter notebooks for data analysis and visualization.
  - `scripts/`: Scripts for training models and other utilities.

## Setup Instructions

1. **Clone the repository**:

   ```
   git clone <repository-url>
   cd ml_space_2
   ```

2. **Install required packages**:
   It is recommended to create a virtual environment and install the necessary packages. You can use `requirements.txt`  or install packages manually:

   ```
   pip install numpy pandas scikit-learn xgboost
   ```

3. **Dataset**:
   Place the dataset in the `diabetes_prediction/data/` directory. Ensure that the dataset is in the correct format as specified in the `data/README.md`.

4. **Running the Models**:
   To train the models, run the `train_model.py` script located in the `diabetes_prediction/scripts/` directory:

   ```
   python diabetes_prediction/scripts/train_model.py
   ```

5. **Exploratory Data Analysis**:
   Use the Jupyter notebook in `diabetes_prediction/notebooks/` for exploratory data analysis:
   ```
   jupyter notebook diabetes_prediction/notebooks/exploratory_data_analysis.ipynb
   ```

## Models Implemented

- Gaussian Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- XGBoost

## Contributors

This project was developed by Kiggundu Sulaiman (Ayman) and Twongiirwe Susan, Computer Science Students at CoCIS of Makerere University Kampala, Uganda.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
