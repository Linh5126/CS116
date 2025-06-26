"""
Setup script để cài đặt environment và kiểm tra dependencies
Chạy script này trước khi sử dụng training system
"""

import subprocess
import sys
import importlib.util
import os

def check_module(module_name, import_name=None):
    """Kiểm tra xem module có được cài đặt không"""
    if import_name is None:
        import_name = module_name
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_requirements():
    """Cài đặt requirements từ file"""
    print("📦 Installing requirements from requirements.txt...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True, check=True)
        
        print("✅ Requirements installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements:")
        print(f"   {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def install_pytorch_manual():
    """Cài đặt PyTorch thủ công với CPU support"""
    print("🔥 Installing PyTorch (CPU version)...")
    
    try:
        # Install PyTorch CPU version
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"
        ], capture_output=True, text=True, check=True)
        
        print("✅ PyTorch installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing PyTorch:")
        print(f"   {e.stderr}")
        return False

def check_all_dependencies():
    """Kiểm tra tất cả dependencies"""
    print("🔍 Checking dependencies...")
    
    dependencies = [
        ("torch", "torch"),
        ("numpy", "numpy"), 
        ("pygame", "pygame"),
        ("matplotlib", "matplotlib"),
        ("opencv-python", "cv2"),
        ("pytmx", "pytmx")  # Thêm pytmx nếu cần
    ]
    
    missing = []
    installed = []
    
    for module_name, import_name in dependencies:
        if check_module(module_name, import_name):
            installed.append(module_name)
            print(f"   ✅ {module_name}")
        else:
            missing.append(module_name)
            print(f"   ❌ {module_name}")
    
    return installed, missing

def test_imports():
    """Test import các modules chính"""
    print("\n🧪 Testing imports...")
    
    test_modules = [
        "torch",
        "numpy", 
        "pygame",
        "matplotlib.pyplot",
        "cv2"
    ]
    
    success = True
    for module in test_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            success = False
    
    return success

def setup_environment():
    """Setup toàn bộ environment"""
    print("🚀 Setting up Hardest Game AI Environment")
    print("=" * 60)
    
    # 1. Kiểm tra Python version
    python_version = sys.version_info
    print(f"🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 7):
        print("⚠️ Warning: Python 3.7+ recommended")
    
    # 2. Kiểm tra pip
    try:
        import pip
        print(f"📦 pip available")
    except ImportError:
        print("❌ pip not available - please install pip first")
        return False
    
    # 3. Check current dependencies
    installed, missing = check_all_dependencies()
    
    if not missing:
        print(f"\n✅ All dependencies already installed!")
        return test_imports()
    
    print(f"\n📋 Missing dependencies: {missing}")
    
    # 4. Install missing dependencies
    if "torch" in missing:
        print("\n🔥 PyTorch not found - installing...")
        if not install_pytorch_manual():
            print("❌ Failed to install PyTorch manually")
            print("💡 Try installing manually:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
            return False
    
    # 5. Install other requirements
    if any(dep != "torch" for dep in missing):
        print("\n📦 Installing other requirements...")
        if not install_requirements():
            print("💡 Try installing manually:")
            print("   pip install -r requirements.txt")
            return False
    
    # 6. Final check
    print("\n🔍 Final verification...")
    return test_imports()

def create_test_script():
    """Tạo script test nhanh"""
    test_content = '''"""
Quick test script để verify environment setup
"""

try:
    import torch
    import numpy as np
    import pygame
    import matplotlib.pyplot as plt
    import cv2
    
    print("✅ All imports successful!")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🔢 NumPy version: {np.__version__}")
    print(f"🎮 Pygame version: {pygame.version.ver}")
    
    # Test PyTorch basic operation
    x = torch.randn(3, 3)
    print(f"🧠 PyTorch test tensor: {x.shape}")
    
    print("🎯 Environment ready for AI training!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Run setup_environment.py first!")
'''
    
    with open('test_environment.py', 'w') as f:
        f.write(test_content)
    
    print("📝 Created test_environment.py")

if __name__ == '__main__':
    print("Hardest Game AI - Environment Setup")
    print("=" * 60)
    print("Choose an option:")
    print("1. Full setup (install all dependencies)")
    print("2. Check current dependencies")
    print("3. Test imports only")
    print("4. Install PyTorch only")
    print("5. Create test script")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        success = setup_environment()
        if success:
            print("\n🎉 Setup completed successfully!")
            print("Now you can run:")
            print("   python test_new_logic.py")
            print("   python test_hard_level.py")
            print("   python train_single.py")
        else:
            print("\n❌ Setup failed - check errors above")
            
    elif choice == '2':
        installed, missing = check_all_dependencies()
        print(f"\n📊 Summary:")
        print(f"   Installed: {len(installed)} modules")
        print(f"   Missing: {len(missing)} modules")
        if missing:
            print(f"   Missing modules: {missing}")
            
    elif choice == '3':
        test_imports()
        
    elif choice == '4':
        install_pytorch_manual()
        
    elif choice == '5':
        create_test_script()
        print("Run: python test_environment.py")
        
    elif choice == '6':
        print("👋 Goodbye!")
        
    else:
        print("❌ Invalid choice") 