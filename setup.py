#!/usr/bin/env python3
"""
Setup script for MTG Rules Assistant.
This script helps users set up their environment and get the app running.
"""

import os
import sys
import subprocess
import getpass

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['streamlit', 'openai', 'scrython']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run: pip install -r requirements.txt")
            return False
    
    return True

def setup_api_key():
    """Set up OpenAI API key."""
    print("\n🔑 OpenAI API Key Setup")
    print("You need an OpenAI API key to use this app.")
    print("Get one at: https://platform.openai.com/api-keys")
    
    # Check if API key is already set
    if os.getenv("OPENAI_API_KEY"):
        print("✅ OPENAI_API_KEY is already set in environment")
        return True
    
    # Ask user for API key
    api_key = getpass.getpass("Enter your OpenAI API key (input will be hidden): ").strip()
    
    if not api_key:
        print("❌ No API key provided. Please set OPENAI_API_KEY environment variable.")
        return False
    
    if not api_key.startswith("sk-"):
        print("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
        return False
    
    # Set environment variable for current session
    os.environ["OPENAI_API_KEY"] = api_key
    print("✅ API key set for current session")
    
    # Ask if user wants to save to .env file
    save_to_env = input("Save API key to .env file for future sessions? (y/n): ").lower().strip()
    if save_to_env in ['y', 'yes']:
        try:
            with open('.env', 'w') as f:
                f.write(f"OPENAI_API_KEY={api_key}\n")
            print("✅ API key saved to .env file")
        except Exception as e:
            print(f"❌ Failed to save .env file: {e}")
    
    return True

def test_openai_connection():
    """Test OpenAI API connection."""
    print("\n🧪 Testing OpenAI API Connection...")
    
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        print("✅ OpenAI API connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🃏 MTG Rules Assistant Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Setup API key
    if not setup_api_key():
        return False
    
    # Test API connection
    if not test_openai_connection():
        return False
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To run the app:")
    print("   streamlit run app.py")
    print("\n📖 For more information, see README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 