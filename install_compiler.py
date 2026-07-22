import os
import sys
import zipfile
import urllib.request
import time

def install_compiler():
    print("==================================================")
    print(" Installing Lightweight C++ Compiler (w64devkit)")
    print("==================================================")
    
    url = "https://github.com/skeeto/w64devkit/releases/download/v1.20.0/w64devkit-1.20.0.zip"
    
    # Extract to User Profile directory to avoid admin permission prompts
    user_profile = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    install_dir = os.path.join(user_profile, "w64devkit")
    zip_path = os.path.join(user_profile, "w64devkit.zip")
    
    print(f"Target Directory: {install_dir}")
    print(f"Downloading from: {url}\n")
    
    start_time = time.time()
    try:
        # Download the zip file
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 # 1 MB chunks
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    
                    elapsed = time.time() - start_time
                    speed = downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rDownloading: {percent:.1f}% | {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB | Speed: {speed:.2f} MB/s")
                    else:
                        sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB | Speed: {speed:.2f} MB/s")
                    sys.stdout.flush()
            print("\nDownload complete!")
            
        # Extract zip file
        print(f"Extracting zip archive to: {user_profile}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(user_profile)
            
        print("\n[Success] C++ Compiler installed successfully!")
        print(f"- Compiler executable: {os.path.join(install_dir, 'bin', 'g++.exe')}")
        print("- You can now run the build script to compile your code.")
        
        # Cleanup zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            
    except Exception as e:
        print(f"\n[Error] Installation failed: {e}")
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass
        sys.exit(1)

if __name__ == "__main__":
    install_compiler()
