#!/usr/bin/env python3
"""
Debug script to check export module
"""

def check_export_module():
    """Check if export module has add_subparser function"""
    try:
        print("🔍 Checking export module...")
        
        # Import the module
        from retfound.cli.commands import export
        
        # Check what attributes it has
        print(f"📋 Export module attributes:")
        attrs = [attr for attr in dir(export) if not attr.startswith('_')]
        for attr in sorted(attrs):
            print(f"  - {attr}")
        
        # Check specifically for add_subparser
        if hasattr(export, 'add_subparser'):
            print("✅ add_subparser function found!")
            print(f"   Type: {type(export.add_subparser)}")
        else:
            print("❌ add_subparser function NOT found!")
        
        # Check file location
        print(f"📁 Module file: {export.__file__}")
        
        # Check if we can read the file content
        with open(export.__file__, 'r') as f:
            content = f.read()
            if 'def add_subparser(' in content:
                print("✅ add_subparser function found in file!")
                # Find the line number
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'def add_subparser(' in line:
                        print(f"   Found at line {i+1}")
                        break
            else:
                print("❌ add_subparser function NOT found in file!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking export module: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_git_status():
    """Check git status"""
    import subprocess
    try:
        print("\n🔍 Checking git status...")
        
        # Check current commit
        result = subprocess.run(['git', 'log', '--oneline', '-1'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"📝 Current commit: {result.stdout.strip()}")
        
        # Check if there are uncommitted changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            if result.stdout.strip():
                print("⚠️  Uncommitted changes found:")
                print(result.stdout)
            else:
                print("✅ No uncommitted changes")
        
    except Exception as e:
        print(f"❌ Error checking git status: {e}")

def main():
    print("🧪 RETFound Export Module Debug")
    print("=" * 40)
    
    success = check_export_module()
    check_git_status()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Debug completed")
    else:
        print("❌ Debug found issues")

if __name__ == "__main__":
    main()
