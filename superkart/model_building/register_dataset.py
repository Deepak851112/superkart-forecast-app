
import os
from huggingface_hub import HfApi, login
from datasets import load_dataset

# Define Hugging Face details
hf_username = "rakesh1248"  # IMPORTANT: Replace with your HF username
dataset_name = "wellness-tourism-package-prediction"

# Get HF_TOKEN from environment variable
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# Log in to Hugging Face
login(hf_token)

# Define the dataset repository ID
repo_id = f"{hf_username}/{dataset_name}"

# Define the path to the local CSV file
csv_file_path = "tourism_project/data/tourism.csv"

# Initialize Hugging Face API
api = HfApi()

print("--------------------------------------------------")
print(f"Attempting to create/verify dataset repository: {repo_id}")
# Create dataset repository (if not exists)
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True
)
print("Dataset repository ready.")

print(f"Attempting to upload file: {csv_file_path} to {repo_id}")
# Upload CSV file
api.upload_file(
    path_or_fileobj=csv_file_path,
    path_in_repo="data.csv",   # Name inside HF repo
    repo_id=repo_id,
    repo_type="dataset"
)
print("CSV uploaded successfully!")

print(f"Verifying dataset by loading from {repo_id}...")
# Verify by loading dataset
dataset = load_dataset(repo_id)
print("Dataset loaded and verified:")
print(dataset)
print("--------------------------------------------------")
