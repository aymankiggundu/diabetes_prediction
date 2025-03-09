# Machine Learning Diabetes Prediction Project Using Electrogram Signal Processig

This project, `Sparse Knowledge Distillation for
Electrogastrogram Signal Processing to
Predict Diabetes`, is focused on predicting diabetes using various machine learning models. The goal is to implement and compare different algorithms to determine their effectiveness in predicting diabetes based on a given dataset under different circustances i.e on original dataset, the on denoised dataset and techniques like Sparse learning, Knowledge Distillation and Sparse Knowledge Distillation we added into the model architectures to try and compare the performances of the models.

## Project Structure

The project is organized into the following directories and files:

- `diabetes_prediction/`: Contains all files related to the diabetes prediction models.
  - `data/`: Directory for dataset-related files.
  - `models/`: Contains implementations of various machine learning models.
  - `notebooks/`: Jupyter notebooks for data analysis and visualization.
  

## Setup Instructions

1. **Clone the repository**:

   ```
   git clone <repository-url>
   cd ml_space_2
   ```

2. **Install required packages**:
   It is recommended to create a virtual environment and install the necessary packages. You can use `requirements.txt` or install packages manually:

   ```
   pip install numpy pandas scikit-learn xgboost
   ```

3. **Dataset**:
   Place the dataset in the `diabetes_prediction/data/` directory. Ensure that the dataset is in the correct format as specified in the `data/README.md`.

4. **Running the Models**:
   To train the models, run each model independently in the models folder and resulst will be appended in the performance csv file as well as on the terminal`/example_model.py` script located in the `diabetes_prediction/models/` directory:

   ```
   python diabetes_prediction/models/example_model.py
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

## Streamlit App

The project includes a Streamlit app for predicting diabetes based on user input. The app can be accessed at the following URL:

[http://13.60.230.84:8501/](http://13.60.230.84:8501/)

## Contributors

This project was developed by Kiggundu Sulaiman (Ayman) and Twongiirwe Susan, Computer Science Students at CoCIS of Makerere University Kampala, Uganda.

## License

This project is licensed under the Makerere University License - see the LICENSE file for details.
