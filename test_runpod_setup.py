#!/usr/bin/env python3
"""
Test RETFound Setup on RunPod
=============================

Quick test script to verify all components work correctly on RunPod.
"""

import sys
import traceback
from pathlib import Path

def test_python_basics():
    """Test Python and basic imports"""
    print("🐍 Testing Python basics...")
    
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        print(f"  ✅ CUDA available: {torch.cuda.is_available()}")
        print(f"  ✅ GPU count: {torch.cuda.device_count()}")
        return True
    except Exception as e:
        print(f"  ❌ PyTorch error: {e}")
        return False

def test_retfound_model():
    """Test RETFound model import"""
    print("🔬 Testing RETFound model...")
    
    try:
        from retfound.models import RETFoundModel
        print("  ✅ RETFound model import OK")
        return True
    except Exception as e:
        print(f"  ❌ RETFound model error: {e}")
        return False

def test_cli():
    """Test CLI functionality"""
    print("⚡ Testing CLI...")
    
    try:
        from retfound.cli.main import create_parser
        parser = create_parser()
        print("  ✅ CLI parser created successfully")
        return True
    except Exception as e:
        print(f"  ❌ CLI error: {e}")
        traceback.print_exc()
        return False

def test_monitoring():
    """Test monitoring components"""
    print("📊 Testing monitoring...")
    
    try:
        from retfound.monitoring.server import create_server
        print("  ✅ Monitoring server import OK")
        
        from retfound.monitoring.data_manager import DataManager
        print("  ✅ Data manager import OK")
        
        return True
    except Exception as e:
        print(f"  ❌ Monitoring error: {e}")
        return False

def test_frontend():
    """Test frontend build"""
    print("🎨 Testing frontend...")
    
    frontend_dir = Path("retfound/monitoring/frontend")
    if not frontend_dir.exists():
        print("  ❌ Frontend directory not found")
        return False
    
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        print("  ❌ package.json not found")
        return False
    
    node_modules = frontend_dir / "node_modules"
    if not node_modules.exists():
        print("  ⚠️  node_modules not found - run ./fix_frontend_permissions.sh")
        return False
    
    print("  ✅ Frontend structure OK")
    return True

def fix_permissions():
    """Fix frontend permissions"""
    print("🔧 Fixing frontend permissions...")
    
    import subprocess
    import os
    
    try:
        # Make script executable
        os.chmod("fix_frontend_permissions.sh", 0o755)
        
        # Run the fix script
        result = subprocess.run(
            ["./fix_frontend_permissions.sh"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode == 0:
            print("  ✅ Frontend permissions fixed")
            return True
        else:
            print(f"  ❌ Fix script failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Permission fix error: {e}")
        return False

def run_quick_demo():
    """Run a quick monitoring demo"""
    print("🚀 Testing monitoring demo...")
    
    try:
        # Import demo components
        from retfound.monitoring.demo import TrainingSimulator
        from retfound.monitoring.server import create_server
        
        print("  ✅ Demo imports successful")
        print("  ℹ️  Full demo available with: python retfound/monitoring/demo.py")
        return True
        
    except Exception as e:
        print(f"  ❌ Demo error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 RETFound RunPod Setup Test")
    print("=" * 40)
    
    tests = [
        ("Python & PyTorch", test_python_basics),
        ("RETFound Model", test_retfound_model),
        ("CLI System", test_cli),
        ("Monitoring", test_monitoring),
        ("Frontend", test_frontend),
        ("Demo System", run_quick_demo),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        results[test_name] = test_func()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n🚀 Ready to start training:")
        print("   python -m retfound.cli train --config configs/runpod.yaml")
        print("\n📊 Or start with monitoring:")
        print("   python retfound/monitoring/server.py &")
        print("   python -m retfound.cli train --config configs/runpod.yaml --monitor")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        
        if not results.get("Frontend", True):
            print("\n🔧 Try fixing frontend:")
            print("   ./fix_frontend_permissions.sh")
        
        if not results.get("CLI System", True):
            print("\n🔧 Try reinstalling dependencies:")
            print("   pip install -r requirements.txt")
        
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)
