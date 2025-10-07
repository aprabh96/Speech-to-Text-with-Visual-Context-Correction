#!/usr/bin/env python3
"""
Groq API Diagnostic Script
Run this on the computer with the Groq issue to diagnose the problem
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv(dotenv_path=os.path.join('data', '.env'))
load_dotenv()

def check_groq_installation():
    """Check if Groq is properly installed"""
    try:
        import groq
        print(f"✅ Groq library imported successfully")
        print(f"   Version: {groq.__version__}")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Groq library: {e}")
        print("   Try: pip install groq")
        return False

def check_groq_api_key():
    """Check if API key is available"""
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"✅ GROQ_API_KEY found (length: {len(api_key)})")
        return api_key
    else:
        print("❌ GROQ_API_KEY not found in environment")
        print("   Check your .env file in the data/ folder")
        return None

def test_groq_client(api_key):
    """Test Groq client initialization"""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        print(f"✅ Groq client created successfully")
        
        # Check available attributes
        attrs = [attr for attr in dir(client) if not attr.startswith('_')]
        print(f"   Available attributes: {attrs}")
        
        # Check audio attribute
        if hasattr(client, 'audio'):
            print(f"✅ Client has 'audio' attribute")
            audio_attrs = [attr for attr in dir(client.audio) if not attr.startswith('_')]
            print(f"   Audio attributes: {audio_attrs}")
            
            # Check transcriptions
            if hasattr(client.audio, 'transcriptions'):
                print(f"✅ Audio has 'transcriptions' attribute")
                trans_attrs = [attr for attr in dir(client.audio.transcriptions) if not attr.startswith('_')]
                print(f"   Transcriptions attributes: {trans_attrs}")
                
                # Check create method
                if hasattr(client.audio.transcriptions, 'create'):
                    print(f"✅ Transcriptions has 'create' method")
                    return True
                else:
                    print(f"❌ Transcriptions missing 'create' method")
            else:
                print(f"❌ Audio missing 'transcriptions' attribute")
        else:
            print(f"❌ Client missing 'audio' attribute")
            
        return False
        
    except Exception as e:
        print(f"❌ Failed to create Groq client: {e}")
        return False

def test_with_small_audio():
    """Test with a minimal audio file if available"""
    try:
        # Look for existing audio files
        recordings_dir = os.path.join("data", "recordings")
        if os.path.exists(recordings_dir):
            audio_files = [f for f in os.listdir(recordings_dir) if f.endswith('.wav')]
            if audio_files:
                print(f"✅ Found audio files: {audio_files}")
                return True
            else:
                print("ℹ️  No audio files found in data/recordings/")
        else:
            print("ℹ️  Recordings directory doesn't exist yet")
        return False
    except Exception as e:
        print(f"⚠️  Error checking audio files: {e}")
        return False

def main():
    """Run all diagnostic checks"""
    print("=" * 60)
    print("GROQ API DIAGNOSTIC SCRIPT")
    print("=" * 60)
    
    print("\n1. Checking Groq library installation...")
    if not check_groq_installation():
        print("\n❌ GROQ LIBRARY NOT INSTALLED")
        print("Run: pip install groq")
        return False
    
    print("\n2. Checking API key...")
    api_key = check_groq_api_key()
    if not api_key:
        print("\n❌ NO API KEY FOUND")
        print("Add GROQ_API_KEY to your data/.env file")
        return False
    
    print("\n3. Testing Groq client...")
    if not test_groq_client(api_key):
        print("\n❌ GROQ CLIENT ISSUE DETECTED")
        print("This is likely a version compatibility issue.")
        print("Try: pip install --upgrade groq")
        return False
    
    print("\n4. Checking for audio files...")
    test_with_small_audio()
    
    print("\n" + "=" * 60)
    print("✅ ALL GROQ CHECKS PASSED!")
    print("The Groq client should work properly.")
    print("If you're still getting errors, it might be a specific")
    print("runtime issue. Check the console output when recording.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nDiagnostic interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during diagnostic: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
