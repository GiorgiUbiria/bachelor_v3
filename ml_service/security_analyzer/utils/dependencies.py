import subprocess
import sys

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'faker',
        'python-dotenv',
        'psutil'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package]) 