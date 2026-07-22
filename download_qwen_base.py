import os
import sys

def download_base_model():
    print("==================================================")
    print(" Downloading Main Base Model: Qwen/Qwen3.6-35B-A3B")
    print("==================================================")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Installing huggingface_hub...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        from huggingface_hub import snapshot_download

    repo_id = "deepseek-ai/DeepSeek-V4-Flash"
    local_dir = "D:/deepseek_raw"
    
    print(f"Repository ID: {repo_id}")
    print(f"Target Directory: {local_dir}")
    print("Downloading weights (~162 GB, running in background)...")
    
    try:
        # Download only safetensors chunks, configs, and ignore other frameworks' files
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.bin", "*.pth", "*.gguf"],
            resume_download=True
        )
        print("\n[Success] Main base model downloaded successfully to D:/qwen_base_raw!")
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")

if __name__ == "__main__":
    download_base_model()
