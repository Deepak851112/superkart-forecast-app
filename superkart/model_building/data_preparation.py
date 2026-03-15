
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login
import numpy as np

# ------------------------------------------------------------
# Configuration parameters for accessing the dataset repository
# ------------------------------------------------------------
hf_username = "deepakietb"  # Update with your Hugging Face account if required
dataset_name = "superkart-sales-forecast-prediction"
repo_id = f"{hf_username}/{dataset_name}"

# ------------------------------------------------------------
# Retrieve authentication token from environment configuration
# ------------------------------------------------------------
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is missing.")

# Authenticate with Hugging Face Hub
login(hf_token)

print("Starting dataset retrieval from Hugging Face Hub...")

# Fetch the dataset from the repository
dataset = load_dataset(repo_id, split='train')

# Convert the dataset object into a pandas DataFrame
df = dataset.to_pandas()

print("Dataset successfully downloaded and converted to DataFrame.")

# ------------------------------------------------------------
# Remove unwanted columns if they are present in the dataset
# ------------------------------------------------------------
df = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')

print("Optional columns 'Unnamed: 0' and 'CustomerID' removed if present.")

# ------------------------------------------------------------
# Separate the feature set and the prediction target column
# ------------------------------------------------------------
X = df.drop('Product_Store_Sales_Total', axis=1)
y = df['Product_Store_Sales_Total']

print("Feature matrix (X) and target variable (y) have been isolated.")

# ------------------------------------------------------------
# Dynamically detect column types for future preprocessing
# ------------------------------------------------------------
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

print(f"Detected numerical feature columns: {numerical_cols}")
print(f"Detected categorical feature columns: {categorical_cols}")

# ------------------------------------------------------------
# Divide the dataset into training and testing subsets
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset partitioned into training and testing segments.")

# ------------------------------------------------------------
# Create directory structure for storing split datasets
# ------------------------------------------------------------
os.makedirs("superkart/model_building/split_data", exist_ok=True)

# Save individual components of the split dataset
X_train.to_csv("superkart/model_building/split_data/X_train.csv", index=False)
X_test.to_csv("superkart/model_building/split_data/X_test.csv", index=False)
y_train.to_csv("superkart/model_building/split_data/y_train.csv", index=False)
y_test.to_csv("superkart/model_building/split_data/y_test.csv", index=False)

print("Separate training and testing files stored locally.")

# ------------------------------------------------------------
# Merge features and targets again for repository upload
# ------------------------------------------------------------
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# File locations for the merged datasets
train_csv_path = "superkart/model_building/split_data/train.csv"
test_csv_path = "superkart/model_building/split_data/test.csv"

# Save combined datasets
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

print("Combined datasets (train.csv and test.csv) prepared for upload.")

# ------------------------------------------------------------
# Upload processed datasets to Hugging Face repository
# ------------------------------------------------------------
api = HfApi()

print("Transferring processed datasets to Hugging Face Hub...")

api.upload_file(
    path_or_fileobj=train_csv_path,
    path_in_repo="train.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=hf_token
)

api.upload_file(
    path_or_fileobj=test_csv_path,
    path_in_repo="test.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=hf_token
)

print("Training and testing dataset files successfully published to Hugging Face Hub.")
