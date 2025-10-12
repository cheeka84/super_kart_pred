# This script registers and uploads your local dataset folder to the Hugging Face Hub.

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Define the target repository ID on Hugging Face Hub.
repo_id = "cheeka84/super-kart-pred"
# Since this script uploads data, we mark it as a dataset repository.
repo_type = "dataset"


# Initialize Hugging Face API client using personal access token already set.
# HF_TOKEN is set as an environment variable
# In GitHub Actions, stored the token securely in repo secrets to load it via os.getenv.
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if repo exists or not

try:
    # Try to fetch information about the specified repo.
    # If it exists, repo_info() will succeed.
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    # If the repo does not exist, create a new one.
    print(f"Space '{repo_id}' not found. Creating new space...")

    # Create a new public dataset repository on Hugging Face.
    # Set private=True if the dataset is confidential.
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    print(f"Space '{repo_id}' created successfully.")


# Step 2: Upload Local Data to Hugging Face

# Upload the entire folder to the Hugging Face Hub.
# - folder_path points to your local dataset directory.
# - repo_id and repo_type specify the destination.

api.upload_folder(
    folder_path="super_kart_pred/data",  # local path to upload
    repo_id=repo_id,                     # target repo on HF Hub
    repo_type=repo_type,                 # repo_type is dataset
)
