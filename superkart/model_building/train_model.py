
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import mlflow
from huggingface_hub import HfApi, login, create_repo
import shutil

# Define Hugging Face details
hf_username = "rakesh1248" # IMPORTANT: Replace with your HF username
dataset_name = "wellness-tourism-package-prediction"
repo_id = f"{hf_username}/{dataset_name}"
model_repo_name = "wellness-tourism-package-prediction-model"

# Get HF_TOKEN from environment variable
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# Log in to Hugging Face
login(hf_token)

# Load the training and testing datasets from Hugging Face Hub
print("Loading training and testing datasets from Hugging Face Hub...")
train_dataset = load_dataset(repo_id, split='train')
train_df = train_dataset.to_pandas()
test_dataset = load_dataset(repo_id, split='test')
test_df = test_dataset.to_pandas()
print("Datasets loaded successfully.")

# Separate features and target variable
X_train = train_df.drop('ProdTaken', axis=1)
y_train = train_df['ProdTaken']
X_test = test_df.drop('ProdTaken', axis=1)
y_test = test_df['ProdTaken']

# Identify categorical and numerical columns dynamically
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

# Preprocessing pipelines for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        solver='liblinear',
        penalty='l1',
        C=0.1
    ))
])

# Model parameters
model_params = {
    'solver': 'liblinear',
    'penalty': 'l1',
    'C': 0.1,
    'random_state': 42
}

# Set the MLflow experiment name
mlflow.set_experiment("Wellness Tourism Package Prediction")

print("Starting MLflow run...")
with mlflow.start_run():
    # Log parameters
    mlflow.log_params(model_params)

    print("Training the model...")
    # Train the model
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc_score", roc_auc)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # Log the model to MLflow Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_regression_model",
        registered_model_name="LogisticRegressionWellnessPackage"
    )
    print("Model and metrics logged to MLflow.")

    # Retrieve the best model from MLflow Model Registry
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions("LogisticRegressionWellnessPackage", stages=None)[0]
    model_uri = latest_version.source
    print(f"Loading model from MLflow URI: {model_uri}")

    # Create a temporary directory to save the model artifacts locally
    temp_model_dir = "./hf_model_export"
    os.makedirs(temp_model_dir, exist_ok=True)

    # Use mlflow.artifacts.download_artifacts to get the model artifacts
    mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path="logistic_regression_model",
        dst_path=temp_model_dir
    )
    print(f"MLflow model artifacts downloaded to: {temp_model_dir}")

    # Create a new model repository on Hugging Face
    repo_id_model = f"{hf_username}/{model_repo_name}"
    create_repo(repo_id=repo_id_model, repo_type="model", exist_ok=True, token=hf_token)
    print(f"Model repository '{model_repo_name}' created or already exists on Hugging Face.")

    # Upload model artifacts to Hugging Face
    api = HfApi()
    print(f"Uploading model artifacts from '{temp_model_dir}' to '{repo_id_model}'...")
    api.upload_folder(
        folder_path=temp_model_dir,
        repo_id=repo_id_model,
        repo_type="model",
        token=hf_token,
        commit_message="Upload latest Logistic Regression model from MLflow"
    )
    print(f"Model '{model_repo_name}' successfully uploaded to Hugging Face Model Hub.")

    # Clean up temporary directory
    shutil.rmtree(temp_model_dir)
    print("Cleaned up temporary model export directory.")

print("Model training and registration process complete.")
