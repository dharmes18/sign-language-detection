import subprocess
import sys
import os

def install_tensorflow_gpu():
    print("🚀 Installing TensorFlow with GPU support for your GTX 1650...")
    print("=" * 60)
    
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("⚠️  Consider creating a virtual environment first:")
        print("   python -m venv sign_language_env")
        print("   sign_language_env\\Scripts\\activate")
        print()
    
    # Install TensorFlow with GPU support
    packages = [
        "tensorflow[and-cuda]==2.16.2",  # TensorFlow with CUDA support
        "nvidia-cudnn-cu12==8.9.7.29",   # cuDNN for GPU acceleration
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    print("\n" + "=" * 60)
    print("Installation completed! Now let's verify GPU detection...")

if __name__ == "__main__":
    install_tensorflow_gpu()