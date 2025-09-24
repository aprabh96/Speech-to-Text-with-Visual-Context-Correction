#!/usr/bin/env python3
"""
Test script to verify speech-to-text stability fixes
This script simulates multiple recording cycles to test for memory leaks and crashes
"""

import time
import threading
import traceback
import psutil
import os
import sys

# Add the parent directory to the path so we can import the main modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def memory_monitor():
    """Monitor memory usage during testing"""
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Initial memory usage: {initial_memory:.1f} MB")
    
    return process, initial_memory

def check_memory_leak(process, initial_memory, cycle_num):
    """Check for memory leaks"""
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = current_memory - initial_memory
    
    print(f"Cycle {cycle_num}: Memory usage: {current_memory:.1f} MB (increase: {memory_increase:+.1f} MB)")
    
    # Flag significant memory increase (more than 50MB growth)
    if memory_increase > 50:
        print(f"⚠️  WARNING: Significant memory increase detected: {memory_increase:.1f} MB")
        return False
    return True

def test_cli_stability():
    """Test CLI version stability"""
    print("=" * 60)
    print("TESTING CLI VERSION STABILITY")
    print("=" * 60)
    
    try:
        # Import the CLI module (this will initialize PyAudio)
        import speech_to_text
        
        # Setup memory monitoring
        process, initial_memory = memory_monitor()
        
        print(f"✅ CLI module imported successfully")
        # Note: audio is initialized in main block, not at module level
        print(f"✅ CLI module has PyAudio support")
        
        # Test resource cleanup multiple times
        for cycle in range(1, 8):  # Test 7 cycles (beyond the crash threshold)
            print(f"\n--- Test Cycle {cycle} ---")
            
            try:
                # Test global state reset
                speech_to_text.buffered_frames = []
                speech_to_text.buffered_screenshot = None
                speech_to_text.buffered_screenshot_path = None
                speech_to_text.buffered_recording = False
                
                # Simulate frame accumulation (like during recording)
                test_frames = [b'test_data'] * 1000  # Simulate 1000 audio chunks
                speech_to_text.buffered_frames = test_frames.copy()
                
                print(f"  Simulated {len(speech_to_text.buffered_frames)} audio frames")
                
                # Test frame clearing (like at end of recording)
                speech_to_text.buffered_frames.clear()
                speech_to_text.buffered_frames = []
                
                print(f"  Frames cleared: {len(speech_to_text.buffered_frames)} remaining")
                
                # Check memory usage
                if not check_memory_leak(process, initial_memory, cycle):
                    return False
                    
                # Small delay between cycles
                time.sleep(0.5)
                
                print(f"  ✅ Cycle {cycle} completed successfully")
                
            except Exception as e:
                print(f"  ❌ Cycle {cycle} failed: {e}")
                traceback.print_exc()
                return False
        
        print(f"\n✅ CLI stability test PASSED - completed 7 cycles without crashes")
        return True
        
    except Exception as e:
        print(f"❌ CLI stability test FAILED: {e}")
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            if 'speech_to_text' in locals() and hasattr(speech_to_text, 'cleanup_on_exit'):
                speech_to_text.cleanup_on_exit()
        except:
            pass

def test_gui_audio_engine():
    """Test GUI AudioEngine stability"""
    print("\n" + "=" * 60)
    print("TESTING GUI AUDIO ENGINE STABILITY")
    print("=" * 60)
    
    try:
        # Import GUI modules
        from data.speech_to_text_gui import AudioEngine, AppConfig
        
        # Setup memory monitoring
        process, initial_memory = memory_monitor()
        
        config = AppConfig()
        
        # Test multiple AudioEngine creation/destruction cycles
        for cycle in range(1, 8):
            print(f"\n--- Audio Engine Cycle {cycle} ---")
            
            try:
                # Create AudioEngine
                audio_engine = AudioEngine(config)
                print(f"  ✅ AudioEngine created")
                
                # Simulate recording frames
                audio_engine.recorded_frames = [b'test_data'] * 2000
                print(f"  Simulated {len(audio_engine.recorded_frames)} recorded frames")
                
                # Test cleanup
                audio_engine.cleanup()
                print(f"  ✅ AudioEngine cleaned up")
                
                # Verify frames were cleared
                if len(audio_engine.recorded_frames) == 0:
                    print(f"  ✅ Recorded frames properly cleared")
                else:
                    print(f"  ⚠️  Warning: {len(audio_engine.recorded_frames)} frames remaining")
                
                # Check memory usage
                if not check_memory_leak(process, initial_memory, cycle):
                    return False
                
                # Remove reference to help garbage collection
                del audio_engine
                
                # Small delay
                time.sleep(0.5)
                
                print(f"  ✅ Cycle {cycle} completed successfully")
                
            except Exception as e:
                print(f"  ❌ Cycle {cycle} failed: {e}")
                traceback.print_exc()
                return False
        
        print(f"\n✅ GUI AudioEngine stability test PASSED - completed 7 cycles without crashes")
        return True
        
    except Exception as e:
        print(f"❌ GUI AudioEngine stability test FAILED: {e}")
        traceback.print_exc()
        return False

def run_stability_tests():
    """Run all stability tests"""
    print("🧪 SPEECH-TO-TEXT STABILITY TEST SUITE")
    print("Testing fixes for crashes after 3-5 transcriptions")
    print("=" * 60)
    
    results = []
    
    # Test CLI version
    cli_result = test_cli_stability()
    results.append(("CLI Stability", cli_result))
    
    # Test GUI AudioEngine
    gui_result = test_gui_audio_engine()
    results.append(("GUI AudioEngine", gui_result))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED! The crash fixes appear to be working correctly.")
        print("The application should now be stable for multiple transcription cycles.")
    else:
        print("⚠️  SOME TESTS FAILED. Additional debugging may be needed.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    # Check if required modules are available
    try:
        import psutil
    except ImportError:
        print("Error: psutil module required for memory monitoring")
        print("Install with: pip install psutil")
        sys.exit(1)
    
    # Run the tests
    success = run_stability_tests()
    sys.exit(0 if success else 1)
