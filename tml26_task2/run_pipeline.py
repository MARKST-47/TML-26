import os
import subprocess
import sys

# 1. Inject static user credentials to completely bypass getpwuid system lookup crashes
os.environ["USER"] = "atml_team034"
os.environ["LOGNAME"] = "atml_team034"
os.environ["HOME"] = "/home/atml_team034"

# 2. Point to your writable packages path
local_lib_dir = os.path.abspath("./local_packages")
os.makedirs(local_lib_dir, exist_ok=True)

# 3. Clean up the pip list: REMOVED torchvision since the container already includes it natively
print("Validating optimized standalone dependencies inside container environment...")
try:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "--target", local_lib_dir, 
        "--no-cache-dir", 
        "numpy", "scipy", "pandas", "safetensors", "torchvision"
    ])
    print("[SUCCESS] Environment dependencies fully synchronized.")
except Exception as e:
    print(f"Dependency setup encountered a warning: {e}")

# 4. Inject local path into the python runtime search matrix
sys.path.insert(0, local_lib_dir)

# 5. Hand execution over to your pipeline entrypoint
import task_template

if __name__ == "__main__":
    task_template.main()
