#!/usr/bin/env python3
"""
Test script to verify CLI fixes are working
"""

def test_config_loading():
    """Test configuration loading"""
    try:
        from retfound.core.config import RETFoundConfig
        from pathlib import Path
        
        print("🔄 Testing configuration loading...")
        config = RETFoundConfig.load(Path('configs/runpod.yaml'))
        print("✅ Config loaded successfully!")
        print(f"📁 Dataset path: {config.data.dataset_path}")
        print(f"🔢 Batch size: {config.training.batch_size}")
        print(f"🎯 Model type: {config.model.type}")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_cli_import():
    """Test CLI imports"""
    try:
        print("\n🔄 Testing CLI imports...")
        from retfound.cli.main import main
        print("✅ CLI imports successful!")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RETFound CLI Fix Verification")
    print("=" * 40)
    
    success = True
    
    # Test config loading
    success &= test_config_loading()
    
    # Test CLI imports
    success &= test_cli_import()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 ALL TESTS PASSED! CLI is ready to use.")
        print("\n🚀 You can now run:")
        print("python -m retfound.cli train --config configs/runpod.yaml --weights oct --modality oct --monitor")
    else:
        print("❌ Some tests failed. Check the errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
