import tkinter as tk
from tkinter import filedialog
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Function to preprocess and train the model
def preprocess_and_train():
    # Load the dataset
    data = pd.read_csv('Feb_2019_ontime.csv')

    # Separate features and target variable
    X = data.drop(columns=['ARR_DEL15'])
    y = data['ARR_DEL15']

    # Analyze data
    print(data.isnull().sum())
    print(data.describe())

    # Preprocessing
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Imputation for target variable
    y = y.fillna(y.median())

    # Imputation and scaling (consider other options based on findings)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fill NaNs in features
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode().iloc[0])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    result_label.config(text="Accuracy: {:.2f}".format(accuracy))

# Create main window
root = tk.Tk()
root.title("Flight Delay Predictor")

# Set window size
root.geometry("400x200")

# Add button to preprocess and train model
predict_button = tk.Button(root, text="Preprocess and Train Model", command=preprocess_and_train)
predict_button.pack(pady=20)

# Add label to display accuracy
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

root.mainloop()
