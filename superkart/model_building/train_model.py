
import os
import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor # Changed from LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Changed to regression metrics
import numpy as np
import mlflow
from huggingface_hub import HfApi, login, create_repo
import shutil

# -------------------------------------------------------------
# Hugging Face configuration values used for dataset and model
# -------------------------------------------------------------
hf_username = "deepakietb" # IMPORTANT: Replace with your HF username
dataset_name = "superkart-sales-forecast-prediction"
repo_id = f"{hf_username}/{dataset_name}"
model_repo_name = "superkart-sales-forecast-prediction" # Ensure this matches your model repo name

# -------------------------------------------------------------
# Read authentication token from environment variables
# -------------------------------------------------------------
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# Authenticate with Hugging Face services
login(hf_token)

# -------------------------------------------------------------
# Retrieve prepared datasets from Hugging Face Hub
# -------------------------------------------------------------
print("Fetching prepared training and testing datasets from Hugging Face repository...")

train_dataset = load_dataset(repo_id, split='train')
test_dataset = load_dataset(repo_id, split='test')

train_df = train_dataset.to_pandas()
test_df = test_dataset.to_pandas()

print("Dataset retrieval finished successfully.")

# -------------------------------------------------------------
# Divide datasets into features and labels
# -------------------------------------------------------------
X_train = train_df.drop('Product_Store_Sales_Total', axis=1) # Updated target column
y_train = train_df['Product_Store_Sales_Total'] # Updated target column

X_test = test_df.drop('Product_Store_Sales_Total', axis=1) # Updated target column
y_test = test_df['Product_Store_Sales_Total'] # Updated target column

# -------------------------------------------------------------
# Automatically detect column types for preprocessing
# -------------------------------------------------------------
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

# -------------------------------------------------------------
# Build preprocessing steps for numerical and categorical data
# -------------------------------------------------------------
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# -------------------------------------------------------------
# Assemble the complete machine learning pipeline
# -------------------------------------------------------------
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor( # Changed to RandomForestRegressor
        random_state=42,
        n_estimators=100,
        max_depth=10
    ))
])

# Parameters recorded for experiment tracking
model_params = {
    'n_estimators': 100,
    'max_depth': 10,
    'random_state': 42
}

# Configure MLflow experiment namespace
mlflow.set_experiment("Superkart Sales Forecast Prediction") # Updated experiment name

print("Initiating MLflow experiment run...")

with mlflow.start_run():

    # Store hyperparameters used for this training session
    mlflow.log_params(model_params)

    print("Commencing model fitting process...")

    model.fit(X_train, y_train.values.ravel()) # Flatten y_train to 1D array

    print("Training phase completed.")

    # ---------------------------------------------------------
    # Produce predictions
    # ---------------------------------------------------------
    y_pred = model.predict(X_test)
    # y_proba is not applicable for regression models

    # ---------------------------------------------------------
    # Compute evaluation metrics
    # ---------------------------------------------------------
    mse = mean_squared_error(y_test.values.ravel(), y_pred) # Flatten y_test for metrics
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test.values.ravel(), y_pred) # Flatten y_test for metrics
    r2 = r2_score(y_test.values.ravel(), y_pred) # Flatten y_test for metrics

    # Log performance indicators to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    print(f"MSE value: {mse:.4f}")
    print(f"RMSE value: {rmse:.4f}")
    print(f"MAE value: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # ---------------------------------------------------------
    # Register trained model within MLflow model registry
    # ---------------------------------------------------------
    mlflow.sklearn.log_model(
        sk_model=model,
        name="random_forest_regressor_model", # Updated artifact path
        registered_model_name="RandomForestSalesForecast" # Updated registered model name
    )

    print("Model artifact and experiment metrics recorded in MLflow.")

    # ---------------------------------------------------------
    # Retrieve the most recent registered model version
    # ---------------------------------------------------------
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions("RandomForestSalesForecast", stages=None)[0] # Updated registered model name

    model_uri = latest_version.source
    print(f"Downloading model artifacts using MLflow URI: {model_uri}")

    # ---------------------------------------------------------
    # Create temporary storage for downloaded model files
    # ---------------------------------------------------------
    temp_model_dir = "./hf_model_export"
    os.makedirs(temp_model_dir, exist_ok=True)

    mlflow.artifacts.download_artifacts(
        run_id=latest_version.run_id,
        artifact_path="random_forest_regressor_model", # Updated artifact path
        dst_path=temp_model_dir
    )

    print(f"MLflow model artifacts downloaded to: {temp_model_dir}")

    # ---------------------------------------------------------
    # Prepare Hugging Face repository for model hosting
    # ---------------------------------------------------------
    repo_id_model = f"{hf_username}/{model_repo_name}"

    create_repo(
        repo_id=repo_id_model,
        repo_type="model",
        exist_ok=True,
        token=hf_token
    )

    print(f"Hugging Face model repository '{model_repo_name}' verified/created.")

    # ---------------------------------------------------------
    # Upload model artifacts to Hugging Face Model Hub
    # ---------------------------------------------------------
    api = HfApi()

    print(f"Publishing model contents from '{temp_model_dir}' to Hugging Face repository '{repo_id_model}'...")

    api.upload_folder(
        folder_path=temp_model_dir,
        repo_id=repo_id_model,
        repo_type="model",
        token=hf_token,
        commit_message="Upload latest RandomForest Regressor model from MLflow" # Updated commit message
    )

    print(f"Model '{model_repo_name}' successfully pushed to Hugging Face.")

    # ---------------------------------------------------------
    # Remove temporary files used during upload process
    # ---------------------------------------------------------
    shutil.rmtree(temp_model_dir)

    print("Temporary artifact directory removed after upload.")

print("Entire workflow for training, tracking, and publishing the model has finished.")
