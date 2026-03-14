
import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, login
import numpy as np

# Define Hugging Face details
hf_username = "rakesh1248"  # IMPORTANT: Replace with your HF username
dataset_name = "wellness-tourism-package-prediction"
repo_id = f"{hf_username}/{dataset_name}"

# Get HF_TOKEN from environment variable
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# Log in to Hugging Face
login(hf_token)

print("Loading dataset from Hugging Face Hub...")
# Load the dataset from Hugging Face Hub
dataset = load_dataset(repo_id, split='train') # Assuming the dataset has a 'train' split

# Convert to pandas DataFrame
df = dataset.to_pandas()
print("Dataset loaded successfully.")

# Drop unnecessary columns, ignoring errors if columns don't exist
df = df.drop(columns=['Unnamed: 0', 'CustomerID'], errors='ignore')
print("Dropped 'Unnamed: 0' and 'CustomerID' columns (if they existed).")

# Separate target variable
X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']
print("Separated features (X) and target (y).")

# Identify categorical and numerical columns dynamically
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in categorical_cols] # Exclude any categorical columns that might be numeric (e.g. CityTier is fine as numeric)
print(f"Identified Numerical Columns: {numerical_cols}")
print(f"Identified Categorical Columns: {categorical_cols}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into training and testing sets.")

# Create a directory to save the split datasets
os.makedirs("tourism_project/model_building/split_data", exist_ok=True)

# Save the training and testing sets locally as individual files
X_train.to_csv("tourism_project/model_building/split_data/X_train.csv", index=False)
X_test.to_csv("tourism_project/model_building/split_data/X_test.csv", index=False)
y_train.to_csv("tourism_project/model_building/split_data/y_train.csv", index=False)
y_test.to_csv("tourism_project/model_building/split_data/y_test.csv", index=False)
print("Individual training and testing sets saved locally.")

# Combine X and y into single dataframes for upload
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Define paths for the new CSV files
train_csv_path = "tourism_project/model_building/split_data/train.csv"
test_csv_path = "tourism_project/model_building/split_data/test.csv"

# Save combined dataframes to CSV
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)
print("Combined training and testing sets (train.csv, test.csv) saved locally.")

# Initialize Hugging Face API
api = HfApi()

print("Uploading training and testing datasets to Hugging Face Hub...")

# Upload train.csv
api.upload_file(
    path_or_fileobj=train_csv_path,
    path_in_repo="train.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=hf_token
)

# Upload test.csv
api.upload_file(
    path_or_fileobj=test_csv_path,
    path_in_repo="test.csv",
    repo_id=repo_id,
    repo_type="dataset",
    token=hf_token
)

print("Training and testing datasets uploaded successfully to Hugging Face Hub.")
