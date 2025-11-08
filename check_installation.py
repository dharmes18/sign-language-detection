import subprocess
import sys

def check_installed_packages():
    print("🔍 Checking installed packages...")
    print("=" * 50)
    
    # Get list of installed packages
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True, capture_output=True)
        packages = result.stdout.lower()
        
        # Check for specific packages
        packages_to_check = ['tensorflow', 'tensorflow-gpu', 'tensorflow-cpu']
        
        found = False
        for pkg in packages_to_check:
            if pkg in packages:
                print(f"✅ {pkg} is installed")
                found = True
                
        if not found:
            print("❌ No TensorFlow packages found in pip list")
            
        # Show all installed packages for debugging
        print("\n📦 All installed packages:")
        print(result.stdout)
        
    except Exception as e:
        print(f"❌ Error checking packages: {e}")

def check_python_path():
    print("\n🐍 Python path information:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Prefix: {sys.prefix}")
    print(f"Base prefix: {sys.base_prefix}")

if __name__ == "__main__":
    check_python_path()
    check_installed_packages()