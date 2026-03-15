
import os
from huggingface_hub import HfApi, login
from datasets import load_dataset

# -------------------------------------------------------------------
# Configuration section defining Hugging Face repository information
# -------------------------------------------------------------------
hf_username = "deepakietb"   # Update with the correct Hugging Face account if necessary
dataset_name = "superkart-sales-forecast-prediction"

# -------------------------------------------------------------------
# Retrieve authentication token from environment variables
# -------------------------------------------------------------------
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("Required environment variable HF_TOKEN is missing.")

# Authenticate with Hugging Face Hub
login(hf_token)

# Build the full dataset repository identifier
repo_id = f"{hf_username}/{dataset_name}"

# Location of the CSV dataset stored locally
csv_file_path = "superkart/data/SuperKart.csv"

# Initialize API interface for interacting with Hugging Face Hub
api = HfApi()

print("==================================================")
print(f"Preparing dataset repository: {repo_id}")

# Create the dataset repository if it does not already exist
api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True
)

print("Repository verification completed.")

print(f"Uploading dataset file '{csv_file_path}' to repository '{repo_id}'")

# Transfer the dataset file into the repository
api.upload_file(
    path_or_fileobj=csv_file_path,
    path_in_repo="data.csv",
    repo_id=repo_id,
    repo_type="dataset"
)

print("Dataset file upload finished successfully.")

print(f"Performing validation by loading dataset from {repo_id}")

# Load the dataset back from Hugging Face to confirm availability
dataset = load_dataset(repo_id)

print("Dataset retrieval successful. Loaded dataset preview:")
print(dataset)
print("==================================================")
