import os
import sys

def download_model():
    print("==================================================")
    print(" Downloading Qwen3.6-35B-A3B-MTP-bf16")
    print("==================================================")
    
    # 1. Install huggingface_hub if missing
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    repo_id = "mlx-community/Qwen3.6-35B-A3B-MTP-bf16"
    local_dir = "D:/qwen_raw"
    
    print(f"Repository ID: {repo_id}")
    print(f"Target Directory: {local_dir}")
    print("Downloading weights (this may take some time depending on your speed)...")
    
    try:
        # Download only the safetensors weights, configs, and tokenizer files
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin", "*.pth"],
            resume_download=True
        )
        print("\n[Success] Model downloaded successfully to D:/qwen_raw!")
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        print("Please check your internet connection and verify that you have space on D: drive.")

if __name__ == "__main__":
    download_model()
