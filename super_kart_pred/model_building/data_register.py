# %%writefile writes this code into the specified file path in a Jupyter/Colab environment
# This script registers and uploads your local dataset folder to the Hugging Face Hub.

from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# -------------------------------
# CONFIGURATION
# -------------------------------

# Define the target repository ID on Hugging Face Hub.
# Format: "<username or org>/<repo_name>"
repo_id = "cheeka84/super-kart-pred"

# Define the repository type. Options: "model", "dataset", or "space".
# Since this script uploads data, we mark it as a dataset repository.
repo_type = "dataset"

# -------------------------------
# AUTHENTICATION & INITIALIZATION
# -------------------------------

# Initialize Hugging Face API client using your personal access token.
# Make sure HF_TOKEN is set as an environment variable before running this script.
# In GitHub Actions, store it securely in repo secrets and load it via os.getenv.
api = HfApi(token=os.getenv("HF_TOKEN"))

# -------------------------------
# STEP 1: CHECK IF THE REPO EXISTS
# -------------------------------

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

# -------------------------------
# STEP 2: UPLOAD LOCAL DATA
# -------------------------------

# Upload the entire folder (recursively) to the Hugging Face Hub.
# - folder_path points to your local dataset directory.
# - repo_id and repo_type specify the destination.
# This operation overwrites any existing files with the same name.
api.upload_folder(
    folder_path="super_kart_pred/data",  # local path to upload
    repo_id=repo_id,                     # target repo on HF Hub
    repo_type=repo_type,                 # ensure correct repo type ("dataset")
)
