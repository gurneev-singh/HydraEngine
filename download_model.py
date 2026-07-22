import os
import sys
import time
import urllib.request

def download_with_progress(url, dest_path):
    print("==================================================")
    print(f" Downloading Base Model: {os.path.basename(dest_path)}")
    print("==================================================")
    print(f"Source URL: {url}")
    print(f"Destination: {dest_path}\n")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    start_time = time.time()
    try:
        # Get content-length header to show percentage
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1 MB chunks
            
            downloaded = 0
            with open(dest_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    # Calculate progress
                    elapsed = time.time() - start_time
                    speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rProgress: {percent:.2f}% | Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB | Speed: {speed:.2f} MB/s | Time: {elapsed:.1f}s")
                    else:
                        sys.stdout.write(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB | Speed: {speed:.2f} MB/s | Time: {elapsed:.1f}s")
                    sys.stdout.flush()
            print("\n\n[Success] Download complete!")
            print(f"- Total Time: {time.time() - start_time:.1f} seconds")
            print(f"- File size: {os.path.getsize(dest_path) / (1024*1024):.1f} MB")
            
    except Exception as e:
        print(f"\n[Error] Download failed: {e}")
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except Exception:
                pass
        sys.exit(1)

if __name__ == "__main__":
    model_url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    
    # Check if D: drive is accessible, else fallback to local directory
    d_drive_path = "D:\\moe_models"
    if os.path.exists("D:\\"):
        dest = os.path.join(d_drive_path, "qwen2.5-3b-instruct-q4_k_m.gguf")
    else:
        print("[Notice] External D: drive not detected. Falling back to local workspace.")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        dest = os.path.join(base_dir, "model", "qwen2.5-3b-instruct-q4_k_m.gguf")
        
    download_with_progress(model_url, dest)
