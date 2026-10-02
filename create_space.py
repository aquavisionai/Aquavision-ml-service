import os
from huggingface_hub import HfApi, login

def create_hf_space():
    token = "***"
    login(token=token, add_to_git_credential=True)
    
    api = HfApi()
    
    user = api.whoami()["name"]
    repo_id = f"{user}/AquaVision-ML"
    print(f"Creating Space: {repo_id}")
    
    try:
        url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True
        )
        print(f"Space created at {url}")
        
        # Also upload the files to avoid git config issues
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=["*.pyc", "__pycache__", ".venv", ".git", "benchmark.py"]
        )
        print("Files uploaded successfully!")
    except Exception as e:
        print(f"Error creating space or uploading: {e}")

if __name__ == "__main__":
    create_hf_space()
