# -*- coding: utf-8 -*-
"""
Professional Speech-to-Text GUI Application with Visual Context Correction
Psynect Corp - www.psynect.ai

A comprehensive GUI application that provides speech-to-text transcription with
screenshot-based context correction using multiple AI models.
"""

import os
import sys
import tempfile
import threading
import time
import wave
import base64
import io
import glob
import json
import queue
import ssl
import atexit
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Get absolute path to script directory at startup (before any os.chdir)
# This ensures paths work even if working directory changes (e.g., Google Drive sync issues after sleep)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global thread exception handler to catch crashes in background threads
def thread_exception_handler(args):
    """Handle uncaught exceptions in threads - ensures they are logged before app exits"""
    print(f"\n❌❌❌ CRITICAL: Thread '{args.thread.name}' crashed! ❌❌❌")
    print(f"Exception type: {args.exc_type.__name__}")
    print(f"Exception value: {args.exc_value}")
    print("Full traceback:")
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    sys.stdout.flush()
    sys.stderr.flush()

# Install the global thread exception handler
threading.excepthook = thread_exception_handler

# GUI Imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import tkinter.font as tkFont
from tkinter import Menu

# Core functionality imports
import numpy as np
import sounddevice as sd  # More reliable than PyAudio
import pyperclip
import soundfile as sf
import keyboard
try:
    import mouse
    MOUSE_AVAILABLE = True
except ImportError:
    MOUSE_AVAILABLE = False
    print("Warning: mouse module not available. Mouse button functionality disabled.")

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from PIL import Image, ImageTk
import pyautogui
import mss
import mss.tools
from anthropic import Anthropic
import websockets
import asyncio
import requests

# Google Gemini SDK
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-genai not installed. Gemini correction disabled. Install with: pip install google-genai")

# Windows API imports
import win32api
import win32gui
import win32con
import ctypes

# System tray imports
try:
    import pystray
    from pystray import MenuItem as item
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False
    print("Warning: pystray not available. System tray functionality disabled.")

# Configuration
from dataclasses import dataclass, asdict

@dataclass
class AppConfig:
    """Application configuration settings"""
    # Audio settings
    rate: int = 24000
    channels: int = 1
    chunk_ms: int = 20
    format: str = "int16"  # Audio format (sounddevice uses string format)
    
    # Recording settings
    hotkey: str = "ctrl+q"
    enable_mouse_button: bool = True  # Middle mouse button recording (enabled by default)
    max_recording_duration: int = 300  # 5 minutes
    max_file_size: int = 25 * 1024 * 1024  # 25 MB
    
    # UI settings
    theme: str = "dark"
    window_width: int = 1000
    window_height: int = 700
    auto_minimize: bool = False
    system_tray: bool = True
    
    # API settings
    openai_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    deepgram_api_key: str = ""
    google_api_key: str = ""

    # Correction settings
    correction_model: str = "gemini"  # "gemini" or "openai" (GPT-4.1)

    # Transcription settings
    backend: int = 1  # 1=GPT-4o, 2=Whisper-1, 3=Groq+correction, 4=Groq only
    mode: int = 1     # 1=high-accuracy, 2=fast, 3=realtime, 4=transcription-only
    use_correction: bool = True
    auto_paste: bool = True
    
    def save_to_file(self, filename: str = "data/app_config.json"):
        """Save configuration to file"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(asdict(self), f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    @classmethod
    def load_from_file(cls, filename: str = "data/app_config.json"):
        """Load configuration from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                return cls(**data)
        except Exception as e:
            print(f"Error loading config: {e}")
        return cls()  # Return default config if loading fails


class AudioEngine:
    """Handles all audio recording using sounddevice (more reliable than PyAudio)"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.recording_active = False
        self.recorded_data = []  # List of numpy arrays
        self.recording_start_time = None
        self.last_error_message = ""
        self._lock = threading.Lock()
        
        # Test audio system
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            print(f"✅ Audio system initialized. Default input: {default_input['name']}")
        except Exception as e:
            print(f"⚠️ Audio system warning: {e}")
            self.last_error_message = str(e)

    def get_sample_width(self) -> int:
        """Return the sample width for 16-bit audio (2 bytes)."""
        return 2  # 16-bit = 2 bytes

    def get_audio_devices(self) -> List[Dict]:
        """Get list of available audio input devices"""
        devices = []
        try:
            all_devices = sd.query_devices()
            for i, device in enumerate(all_devices):
                if device['max_input_channels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device['name'],
                        'channels': device['max_input_channels'],
                        'sample_rate': device['default_samplerate']
                    })
        except Exception as e:
            print(f"Error getting audio devices: {e}")
        return devices

    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """Start audio recording using sounddevice"""
        print("🎙️ Attempting to start recording...")
        sys.stdout.flush()
        
        if self.recording_active:
            print("⚠️ Recording already active, skipping start")
            return False

        try:
            # Clear previous recording data
            with self._lock:
                self.recorded_data = []
            
            self.recording_start_time = time.time()
            self.recording_active = True
            
            print(f"🎙️ Starting audio stream: rate={self.config.rate}, channels={self.config.channels}, device={device_index}")
            sys.stdout.flush()

            # Start recording in a background thread using sounddevice's InputStream
            def audio_callback(indata, frames, time_info, status):
                """Callback function called by sounddevice for each audio block"""
                if status:
                    print(f"⚠️ Audio status: {status}")
                    sys.stdout.flush()
                
                if self.recording_active:
                    # Make a copy of the data
                    with self._lock:
                        self.recorded_data.append(indata.copy())

            # Create and start the input stream
            self.stream = sd.InputStream(
                samplerate=self.config.rate,
                channels=self.config.channels,
                dtype='int16',  # 16-bit audio
                device=device_index,
                callback=audio_callback,
                blocksize=int(self.config.rate * self.config.chunk_ms / 1000)
            )
            self.stream.start()
            
            print("✅ Audio stream started successfully")
            sys.stdout.flush()

            # Start a monitoring thread for duration limit and status
            threading.Thread(target=self._monitor_recording, daemon=True).start()
            
            return True

        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            traceback.print_exc()
            sys.stdout.flush()
            self.last_error_message = str(e)
            self.recording_active = False
            return False

    def _monitor_recording(self):
        """Monitor recording duration and provide status updates"""
        last_status_time = time.time()
        status_interval = 5.0
        
        while self.recording_active:
            try:
                current_time = time.time()
                duration = current_time - self.recording_start_time if self.recording_start_time else 0
                
                # Check max duration
                if duration > self.config.max_recording_duration:
                    print(f"⏰ Recording stopped: max duration ({self.config.max_recording_duration}s) reached")
                    sys.stdout.flush()
                    self.recording_active = False
                    break
                
                # Periodic status update
                if current_time - last_status_time >= status_interval:
                    with self._lock:
                        chunk_count = len(self.recorded_data)
                    print(f"🎙️ Recording status: {chunk_count} chunks captured ({duration:.1f}s)")
                    sys.stdout.flush()
                    last_status_time = current_time
                
                time.sleep(0.1)  # Small sleep to avoid busy-waiting
                
            except Exception as e:
                print(f"❌ Error in recording monitor: {e}")
                traceback.print_exc()
                sys.stdout.flush()
                break
        
        # Final status
        with self._lock:
            final_chunk_count = len(self.recorded_data)
        print(f"🎙️ Recording monitor ended. Total chunks captured: {final_chunk_count}")
        if final_chunk_count == 0:
            print("❌ WARNING: No audio was captured! Check your microphone.")
        sys.stdout.flush()

    def stop_recording(self) -> Tuple[bool, float]:
        """Stop audio recording and return success status and duration"""
        was_active = self.recording_active
        self.recording_active = False
        
        duration = 0.0
        if self.recording_start_time:
            duration = time.time() - self.recording_start_time

        # Stop the stream
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
        
        # Check if we have data
        with self._lock:
            has_data = len(self.recorded_data) > 0
        
        if not has_data and not was_active:
            return False, 0.0
            
        return has_data, duration

    @property
    def recorded_frames(self):
        """Legacy property - returns recorded data as bytes for compatibility"""
        with self._lock:
            if not self.recorded_data:
                return []
            # Convert numpy arrays to bytes
            return [chunk.tobytes() for chunk in self.recorded_data]

    def get_fallback_documents_dir(self):
        """Get the user's Documents folder as fallback location."""
        try:
            if os.name == 'nt':
                documents_path = os.path.join(os.path.expanduser("~"), "Documents", "SpeechToText_Recordings")
            else:
                documents_path = os.path.join(os.path.expanduser("~"), "Documents", "SpeechToText_Recordings")
            
            os.makedirs(documents_path, exist_ok=True)
            return documents_path
        except Exception as e:
            print(f"Error accessing Documents folder: {e}")
            import tempfile
            return tempfile.gettempdir()

    def save_recording(self, filename: str) -> bool:
        """Save recorded audio to file using soundfile (more reliable than wave)"""
        with self._lock:
            if not self.recorded_data:
                return False
            
            try:
                # Concatenate all recorded chunks
                audio_data = np.concatenate(self.recorded_data, axis=0)
                
                # Save using soundfile
                sf.write(filename, audio_data, self.config.rate)
                return True
            except Exception as e:
                print(f"Error saving recording: {e}")
                traceback.print_exc()
                return False

    def safe_save_recording(self, primary_filename: str, backup_filename: str = None) -> tuple:
        """
        Safely saves recorded audio with fallback mechanisms.
        """
        with self._lock:
            if not self.recorded_data:
                return False, None, None
            
            try:
                # Concatenate all recorded chunks
                audio_data = np.concatenate(self.recorded_data, axis=0)
            except Exception as e:
                print(f"❌ Failed to concatenate audio data: {e}")
                return False, None, None
        
        def _save_to_file(filename, data):
            """Internal function to save numpy array to WAV file."""
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                sf.write(filename, data, self.config.rate)
                
                if os.path.exists(filename) and os.path.getsize(filename) > 100:
                    return True
                return False
            except Exception as e:
                print(f"Failed to save to {filename}: {e}")
                traceback.print_exc()
                return False
        
        # Try primary location first
        primary_success = False
        try:
            if _save_to_file(primary_filename, audio_data):
                primary_success = True
                print(f"✅ Primary save successful: {primary_filename}")
            else:
                print(f"❌ Primary save failed: {primary_filename}")
        except Exception as e:
            print(f"❌ Primary save error: {e}")
        
        # If primary failed, try Documents folder fallback
        fallback_success = False
        fallback_filename = None
        if not primary_success:
            try:
                fallback_dir = self.get_fallback_documents_dir()
                fallback_filename = os.path.join(fallback_dir, os.path.basename(primary_filename))
                
                if _save_to_file(fallback_filename, audio_data):
                    fallback_success = True
                    print(f"✅ FALLBACK save successful: {fallback_filename}")
                    print(f"⚠️  Primary location failed - audio saved to Documents folder")
                else:
                    print(f"❌ Fallback save failed: {fallback_filename}")
            except Exception as e:
                print(f"❌ Fallback save error: {e}")
        
        # Try to save backup copy (timestamped)
        backup_success = False
        final_backup_filename = backup_filename
        if backup_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if primary_success:
                backup_dir = os.path.dirname(primary_filename)
            else:
                backup_dir = self.get_fallback_documents_dir()
            final_backup_filename = os.path.join(backup_dir, f"recording_{timestamp}.wav")
        
        if final_backup_filename:
            try:
                if _save_to_file(final_backup_filename, audio_data):
                    backup_success = True
                    print(f"✅ Backup save successful: {final_backup_filename}")
                else:
                    print(f"❌ Backup save failed: {final_backup_filename}")
            except Exception as e:
                print(f"❌ Backup save error: {e}")
        
        # Determine what to return
        if primary_success:
            return True, primary_filename, final_backup_filename if backup_success else None
        elif fallback_success:
            return True, fallback_filename, final_backup_filename if backup_success else None
        else:
            print("🚨 CRITICAL: All save attempts failed!")
            return False, None, None
    
    def cleanup(self):
        """Clean up audio resources"""
        self.recording_active = False
        time.sleep(0.1)
        
        with self._lock:
            self.recorded_data = []
        
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error cleaning up audio stream: {e}")
            finally:
                self.stream = None

    def recover_after_error(self) -> bool:
        """Reset the audio engine after an error."""
        self.cleanup()
        self.last_error_message = ""
        return True

    def ensure_audio_ready(self) -> bool:
        """Check if audio system is ready."""
        try:
            sd.query_devices()
            return True
        except Exception as e:
            self.last_error_message = str(e)
            return False


class TranscriptionEngine:
    """Handles transcription using various AI services"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.openai_client = None
        self.groq_client = None
        self.claude_client = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize AI service clients"""
        # Load environment variables: prefer data/.env, then root .env
        try:
            load_dotenv(dotenv_path=os.path.join('data', '.env'))
        except Exception:
            pass
        load_dotenv()
        
        # Update config with environment variables if config values are empty
        if not self.config.openai_api_key:
            self.config.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.config.groq_api_key:
            self.config.groq_api_key = os.getenv("GROQ_API_KEY", "")
        if not self.config.anthropic_api_key:
            self.config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not hasattr(self.config, 'deepgram_api_key') or not self.config.deepgram_api_key:
            self.config.deepgram_api_key = os.getenv("DEEPGRAM_API_KEY", "")
        if not hasattr(self.config, 'google_api_key') or not self.config.google_api_key:
            self.config.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        
        # OpenAI
        api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.openai_client = OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client: {e}")
        
        # Groq
        api_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY")
        if api_key:
            try:
                self.groq_client = Groq(api_key=api_key)
                
                # Check version compatibility
                try:
                    import groq
                    version = getattr(groq, '__version__', 'unknown')
                    print(f"Groq library version detected: {version}")
                    
                    # Test the client to ensure it has the required audio.transcriptions attribute
                    if not (hasattr(self.groq_client, 'audio') and hasattr(self.groq_client.audio, 'transcriptions')):
                        print(f"⚠️  Groq version {version} detected - audio transcription not supported.")
                        print("   Groq features will be disabled. App will use OpenAI only.")
                        print("   (Audio transcription requires Groq v0.20.0+)")
                        self.groq_client = None
                    else:
                        print("✅ Groq client initialized successfully with audio transcription support.")
                except Exception as version_check_error:
                    print(f"Note: Could not verify Groq version compatibility: {version_check_error}")
                    self.groq_client = None
                    
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
                self.groq_client = None
        
        # Anthropic (Claude)
        api_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            try:
                self.claude_client = Anthropic(api_key=api_key)
            except Exception as e:
                print(f"Failed to initialize Claude client: {e}")
    
    def get_available_services(self) -> Dict[str, bool]:
        """Get status of available transcription services"""
        return {
            'openai': self.openai_client is not None,
            'groq': self.groq_client is not None,
            'claude': self.claude_client is not None,
            'deepgram': bool(getattr(self.config, 'deepgram_api_key', ''))
        }
    
    def transcribe_file(self, file_path: str, progress_callback=None) -> Tuple[bool, str]:
        """Transcribe audio file using selected backend"""
        try:
            if progress_callback:
                progress_callback("Starting transcription...", 10)
            
            if self.config.backend in [1, 2] and self.openai_client:
                return self._transcribe_openai(file_path, progress_callback)
            elif self.config.backend in [3, 4] and self.groq_client:
                return self._transcribe_groq(file_path, progress_callback)
            elif self.config.backend in [5, 6, 7] and getattr(self.config, 'deepgram_api_key', ''):
                return self._transcribe_deepgram(file_path, progress_callback)
            else:
                return False, "No suitable transcription service available"
                
        except Exception as e:
            return False, f"Transcription error: {e}"
    
    def _transcribe_openai(self, file_path: str, progress_callback=None) -> Tuple[bool, str]:
        """Transcribe using OpenAI"""
        try:
            if progress_callback:
                progress_callback("Transcribing with OpenAI...", 50)
                
            model = "whisper-1" if self.config.backend == 2 else "gpt-4o-transcribe"
            
            with open(file_path, "rb") as audio_file:
                transcription = self.openai_client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="text"
                )
            
            if progress_callback:
                progress_callback("Transcription completed", 100)
                
            return True, str(transcription)
            
        except Exception as e:
            return False, f"OpenAI transcription failed: {e}"
    def _transcribe_groq(self, file_path: str, progress_callback=None) -> Tuple[bool, str]:
        """Transcribe using Groq"""
        try:
            print("🔍 DEBUG: Starting Groq transcription...")
            if progress_callback:
                progress_callback("Transcribing with Groq...", 50)
            
            print(f"🔍 DEBUG: Opening audio file: {file_path}")
            with open(file_path, "rb") as file:
                # Debug Groq client state
                if not self.groq_client:
                    raise Exception("Groq client is None")
                if not hasattr(self.groq_client, 'audio'):
                    raise Exception(f"Groq client missing 'audio' attribute. Available: {[attr for attr in dir(self.groq_client) if not attr.startswith('_')]}")
                if not hasattr(self.groq_client.audio, 'transcriptions'):
                    raise Exception(f"Groq audio missing 'transcriptions' attribute. Available: {[attr for attr in dir(self.groq_client.audio) if not attr.startswith('_')]}")
                
                print("🔍 DEBUG: Reading file content...")
                file_content = file.read()
                print(f"🔍 DEBUG: File content size: {len(file_content)} bytes")
                
                print("🔍 DEBUG: Calling Groq API...")
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), file_content),
                    model="whisper-large-v3",
                    response_format="text",
                    language="en"
                )
                print("🔍 DEBUG: Groq API call completed successfully")
            
            if progress_callback:
                progress_callback("Transcription completed", 100)
                
            # Handle different response formats
            if isinstance(transcription, str):
                print(f"🔍 DEBUG: Got string response, length: {len(transcription)}")
                return True, transcription
            elif hasattr(transcription, 'text'):
                print(f"🔍 DEBUG: Got object response, text length: {len(transcription.text)}")
                return True, transcription.text
            else:
                print(f"🔍 DEBUG: Got unknown response type: {type(transcription)}")
                return True, str(transcription)
                
        except Exception as e:
            print(f"❌ DEBUG: Groq transcription error: {e}")
            traceback.print_exc()
            return False, f"Groq transcription failed: {e}"

    def _transcribe_deepgram(self, file_path: str, progress_callback=None) -> Tuple[bool, str]:
        """Transcribe using Deepgram"""
        try:
            print("🔍 DEBUG: Starting Deepgram transcription...")
            if progress_callback:
                progress_callback("Transcribing with Deepgram...", 50)
            
            api_key = getattr(self.config, 'deepgram_api_key', '')
            if not api_key:
                return False, "Deepgram API key not configured."
            
            # Target flux model on backend=7, nova-3 on others
            if self.config.backend == 7:
                model = "flux"
            else:
                model = "nova-3"
                
            print(f"🔍 DEBUG: Deepgram selected model: {model}")

            url = f"https://api.deepgram.com/v1/listen?model={model}&smart_format=true"
            headers = {
                "Authorization": f"Token {api_key}"
            }
            
            with open(file_path, "rb") as audio_file:
                response = requests.post(url, headers=headers, data=audio_file)
            
            response.raise_for_status()
            data = response.json()
            
            if progress_callback:
                progress_callback("Transcription completed", 100)
                
            transcript = ""
            if "results" in data and "channels" in data["results"] and len(data["results"]["channels"]) > 0:
                alternatives = data["results"]["channels"][0].get("alternatives", [])
                if len(alternatives) > 0:
                    transcript = alternatives[0].get("transcript", "")
                    
            return True, transcript
                
        except Exception as e:
            print(f"❌ DEBUG: Deepgram transcription error: {e}")
            traceback.print_exc()
            return False, f"Deepgram transcription failed: {e}"


class ScreenshotEngine:
    """Handles screenshot capture and vision-based correction"""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    def capture_screenshot(self) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Capture screenshot of current screen"""
        try:
            mouse_x, mouse_y = pyautogui.position()
            
            with mss.mss() as sct:
                monitors = sct.monitors
                
                # Find monitor containing mouse cursor
                target_monitor = next(
                    (m for i, m in enumerate(monitors) if i > 0 and
                     m["left"] <= mouse_x < m["left"] + m["width"] and
                     m["top"] <= mouse_y < m["top"] + m["height"]),
                    monitors[1] if len(monitors) > 1 else monitors[0]
                )
                
                screenshot_mss = sct.grab(target_monitor)
                img = Image.frombytes("RGB", screenshot_mss.size, screenshot_mss.bgra, "raw", "BGRX")
                
                # Save to script folder (using absolute path for reliability)
                screenshots_dir = os.path.join(SCRIPT_DIR, "screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)
                temp_path = os.path.join(screenshots_dir, "latest_screenshot.jpg")
                img.save(temp_path)
                
                return img, temp_path
                
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None, None


class CorrectionEngine:
    """Handles transcription correction using vision models"""
    
    # System prompt for all correction models
    CORRECTION_SYSTEM_PROMPT = """You are a transcription corrector. You receive a speech-to-text transcript and a screenshot of the user's screen.

TASK: Fix transcription errors using the screenshot as context. Return ONLY the corrected transcript.

RULES:
1. Match proper nouns, technical terms, variable names, file paths, URLs, and UI labels visible on screen — these are the most likely transcription errors.
2. Fix homophones and misheard words (e.g., "their" vs "there", "Cooper Netties" → "Kubernetes" if visible on screen).
3. Fix obvious punctuation and sentence boundaries that the STT model missed.
4. Do NOT add, remove, or rephrase content. Preserve the speaker's exact wording and style.
5. Do NOT describe or transcribe the screenshot itself.
6. If the transcript is empty, return: No transcription provided

Output the corrected transcript only. No quotes, no explanation."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.openai_client = None
        self.claude_client = None
        
        # Initialize OpenAI
        if config.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=config.openai_api_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI client for correction: {e}")
        
        # Initialize Claude
        if config.anthropic_api_key:
            try:
                self.claude_client = Anthropic(api_key=config.anthropic_api_key)
            except Exception as e:
                print(f"Failed to initialize Claude client for correction: {e}")

        # Initialize Gemini
        self.gemini_client = None
        if GEMINI_AVAILABLE and config.google_api_key:
            try:
                self.gemini_client = genai.Client(api_key=config.google_api_key)
                print("✅ Gemini client initialized for correction.")
            except Exception as e:
                print(f"Failed to initialize Gemini client for correction: {e}")

    def encode_image_to_base64(self, image: Image.Image) -> Optional[str]:
        """Convert PIL Image to base64"""
        if image is None:
            return None
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding image to base64: {e}")
            return None
    
    def correct_with_openai(self, transcription: str, image: Image.Image) -> str:
        """Correct transcription using OpenAI GPT-4.1 with vision"""
        if not self.openai_client:
            print("OpenAI correction not available; skipping.")
            return transcription
        
        if image is None:
            print("No screenshot available for OpenAI correction.")
            return transcription
        
        try:
            # Encode screenshot as base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
            
            # Choose model based on selected transcription backend
            model = "gpt-4o-mini" if getattr(self.config, 'backend', 1) == 2 else "gpt-4.1"
            print(f"Using {model} for correction...")
            
            resp = self.openai_client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": self.CORRECTION_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"Here is the raw transcription:\n\"{transcription}\""},
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}
                        ]
                    }
                ]
            )
            return resp.output_text.strip()
        except Exception as e:
            print(f"OpenAI correction failed: {e}. Trying Claude fallback.")
            return self.correct_with_claude(transcription, image)
    
    def correct_with_claude(self, transcription: str, image: Image.Image) -> str:
        """Correct transcription using Claude vision"""
        if not self.claude_client:
            print("Claude correction not available.")
            return transcription
        
        if image is None:
            print("No screenshot available for Claude correction.")
            return transcription
        
        try:
            base64_image = self.encode_image_to_base64(image)
            if not base64_image:
                return transcription
            
            print("Sending request to Claude for correction...")
            response = self.claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=3000,
                system=self.CORRECTION_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here's the transcribed speech: \"{transcription}\""},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64_image}}
                    ]
                }]
            )
            
            # Handle potential differences in response structure
            if response.content and isinstance(response.content, list) and hasattr(response.content[0], 'text'):
                corrected_text = response.content[0].text
            else:
                print(f"Warning: Unexpected Claude response structure: {response}")
                corrected_text = transcription
            
            print(f"Original: {transcription}")
            print(f"Corrected: {corrected_text}")
            return corrected_text
        except Exception as e:
            print(f"Error during Claude correction: {e}")
            return transcription
    
    def correct_with_gemini(self, transcription: str, image: Image.Image) -> str:
        """Correct transcription using Gemini 3 Flash with vision"""
        if not self.gemini_client:
            print("Gemini correction not available; skipping.")
            return transcription

        if image is None:
            print("No screenshot available for Gemini correction.")
            return transcription

        try:
            print("Using Gemini 3 Flash for correction...")
            response = self.gemini_client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=[image, f'Here is the raw transcription:\n"{transcription}"'],
                config=genai_types.GenerateContentConfig(
                    system_instruction=self.CORRECTION_SYSTEM_PROMPT,
                    max_output_tokens=3000,
                    temperature=0.1,
                )
            )
            corrected = response.text.strip()
            print(f"Original: {transcription}")
            print(f"Corrected: {corrected}")
            return corrected
        except Exception as e:
            print(f"Gemini correction failed: {e}. Trying fallback.")
            # Fall back to OpenAI then Claude
            if self.openai_client:
                return self.correct_with_openai(transcription, image)
            return self.correct_with_claude(transcription, image)

    def apply_correction(self, transcription: str, image: Image.Image) -> str:
        """Apply correction using available vision models based on user preference"""
        if not transcription.strip():
            return transcription

        preferred = getattr(self.config, 'correction_model', 'gemini')

        if preferred == "gemini" and self.gemini_client:
            return self.correct_with_gemini(transcription, image)
        elif preferred == "openai" and self.openai_client:
            return self.correct_with_openai(transcription, image)
        # Fallback chain: try any available service
        elif self.gemini_client:
            return self.correct_with_gemini(transcription, image)
        elif self.openai_client:
            return self.correct_with_openai(transcription, image)
        elif self.claude_client:
            return self.correct_with_claude(transcription, image)
        else:
            print("No correction services available.")
            return transcription


class ThemeManager:
    """Manages application themes and styling"""
    
    THEMES = {
        'dark': {
            'bg': '#2b2b2b',
            'fg': '#ffffff',
            'select_bg': '#404040',
            'select_fg': '#ffffff',
            'button_bg': '#404040',
            'button_fg': '#ffffff',
            'button_active_bg': '#505050',
            'button_active_fg': '#ffffff',
            'entry_bg': '#404040',
            'entry_fg': '#ffffff',
            'entry_insert_bg': '#ffffff',
            'text_bg': '#1e1e1e',
            'text_fg': '#ffffff',
            'menu_bg': '#2b2b2b',
            'menu_fg': '#ffffff',
            'frame_bg': '#2b2b2b',
            'label_bg': '#2b2b2b',
            'label_fg': '#ffffff'
        },
        'light': {
            'bg': '#f0f0f0',
            'fg': '#000000',
            'select_bg': '#e0e0e0',
            'select_fg': '#000000',
            'button_bg': '#e0e0e0',
            'button_fg': '#000000',
            'button_active_bg': '#d0d0d0',
            'button_active_fg': '#000000',
            'entry_bg': '#ffffff',
            'entry_fg': '#000000',
            'entry_insert_bg': '#000000',
            'text_bg': '#ffffff',
            'text_fg': '#000000',
            'menu_bg': '#f0f0f0',
            'menu_fg': '#000000',
            'frame_bg': '#f0f0f0',
            'label_bg': '#f0f0f0',
            'label_fg': '#000000'
        },
        'professional': {
            'bg': '#f8f9fa',
            'fg': '#212529',
            'select_bg': '#e9ecef',
            'select_fg': '#212529',
            'button_bg': '#007bff',
            'button_fg': '#ffffff',
            'button_active_bg': '#0056b3',
            'button_active_fg': '#ffffff',
            'entry_bg': '#ffffff',
            'entry_fg': '#212529',
            'entry_insert_bg': '#212529',
            'text_bg': '#ffffff',
            'text_fg': '#212529',
            'menu_bg': '#f8f9fa',
            'menu_fg': '#212529',
            'frame_bg': '#f8f9fa',
            'label_bg': '#f8f9fa',
            'label_fg': '#212529'
        }
    }
    
    @classmethod
    def apply_theme(cls, root: tk.Tk, theme_name: str):
        """Apply theme to the application"""
        if theme_name not in cls.THEMES:
            theme_name = 'dark'
            
        theme = cls.THEMES[theme_name]
        
        # Configure ttk styles
        style = ttk.Style()
        # Use a theme that honors custom styling (especially on Windows)
        try:
            if theme_name == 'dark':
                # clam tends to respect bg/fg customizations better than vista/xpnative
                style.theme_use('clam')
        except Exception:
            pass
        
        # Configure main window
        root.configure(bg=theme['bg'])
        
        # Configure ttk widgets with more comprehensive theming
        style.configure('TFrame', background=theme['bg'])
        style.configure('TLabel', background=theme['bg'], foreground=theme['fg'])
        style.configure('TLabelFrame', background=theme['bg'], foreground=theme['fg'])
        style.configure('TLabelFrame.Label', background=theme['bg'], foreground=theme['fg'])
        
        # Button styling - Force dark theme colors
        if theme_name == 'dark':
            style.configure('TButton', 
                           background='#404040', 
                           foreground='#ffffff',
                           borderwidth=1,
                           focuscolor='none',
                           relief='raised')
            style.map('TButton',
                     background=[('active', '#505050'),
                               ('pressed', '#505050'),
                               ('!disabled', '#404040')],
                     foreground=[('active', '#ffffff'),
                               ('pressed', '#ffffff'),
                               ('!disabled', '#ffffff')])
        else:
            style.configure('TButton', 
                           background=theme['button_bg'], 
                           foreground=theme['button_fg'],
                           borderwidth=1,
                           focuscolor='none')
            style.map('TButton',
                     background=[('active', theme.get('button_active_bg', theme['button_bg'])),
                               ('pressed', theme.get('button_active_bg', theme['button_bg']))],
                     foreground=[('active', theme.get('button_active_fg', theme['button_fg'])),
                               ('pressed', theme.get('button_active_fg', theme['button_fg']))])
        
        # Entry and Combobox styling
        style.configure('TEntry', 
                       fieldbackground=theme['entry_bg'], 
                       foreground=theme['entry_fg'],
                       insertcolor=theme.get('entry_insert_bg', theme['entry_fg']),
                       borderwidth=1)
        style.configure('TCombobox', 
                       fieldbackground=theme['entry_bg'], 
                       foreground=theme['entry_fg'],
                       insertcolor=theme.get('entry_insert_bg', theme['entry_fg']),
                       borderwidth=1)
        style.configure('TSpinbox', 
                       fieldbackground=theme['entry_bg'], 
                       foreground=theme['entry_fg'],
                       insertcolor=theme.get('entry_insert_bg', theme['entry_fg']),
                       borderwidth=1)
        
        # Checkbutton and Radiobutton styling
        style.configure('TCheckbutton', 
                       background=theme['bg'], 
                       foreground=theme['fg'],
                       focuscolor='none')
        style.configure('TRadiobutton', 
                       background=theme['bg'], 
                       foreground=theme['fg'],
                       focuscolor='none')
        
        # Notebook styling
        style.configure('TNotebook', background=theme['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', 
                       background=theme['select_bg'], 
                       foreground=theme['select_fg'],
                       padding=[12, 8])
        style.map('TNotebook.Tab',
                 background=[('selected', theme['bg']),
                           ('active', theme['button_bg'])],
                 foreground=[('selected', theme['fg']),
                           ('active', theme['button_fg'])])
        
        # Progressbar styling
        style.configure('TProgressbar', background=theme['button_bg'])
        
        # Treeview styling
        style.configure('Treeview', 
                       background=theme['text_bg'],
                       foreground=theme['text_fg'],
                       fieldbackground=theme['text_bg'])
        style.configure('Treeview.Heading',
                       background=theme['button_bg'],
                       foreground=theme['button_fg'])
        
        return theme


class HotkeyManager:
    """Manages global hotkeys for the application"""
    
    def __init__(self):
        self.registered_hotkeys = {}
        self.mouse_listener = None
    
    def register_hotkey(self, hotkey: str, callback):
        """Register a global hotkey"""
        try:
            if hotkey in self.registered_hotkeys:
                keyboard.remove_hotkey(self.registered_hotkeys[hotkey])
            
            hook_id = keyboard.add_hotkey(hotkey, callback)
            self.registered_hotkeys[hotkey] = hook_id
            return True
        except Exception as e:
            print(f"Error registering hotkey {hotkey}: {e}")
            return False
    
    def register_mouse_button(self, button: str, callback):
        """Register mouse button listener"""
        if not MOUSE_AVAILABLE:
            print("Mouse functionality not available")
            return False
            
        try:
            if self.mouse_listener:
                mouse.unhook(self.mouse_listener)
            
            if button == "middle":
                self.mouse_listener = mouse.on_middle_click(callback)
            return True
        except Exception as e:
            print(f"Error registering mouse button {button}: {e}")
            return False
    
    def unregister_all(self):
        """Unregister all hotkeys and mouse listeners"""
        try:
            keyboard.unhook_all()
            if self.mouse_listener and MOUSE_AVAILABLE:
                mouse.unhook(self.mouse_listener)
            self.registered_hotkeys.clear()
            self.mouse_listener = None
        except Exception as e:
            print(f"Error unregistering hotkeys: {e}")


class SystemTrayManager:
    """Manages system tray integration"""
    
    def __init__(self, app_instance):
        self.app = app_instance
        self.tray_icon = None
        self.is_visible = True
        
        if not PYSTRAY_AVAILABLE:
            return
        
        # Create tray icon
        self._create_tray_icon()
    
    def _create_tray_icon(self):
        """Create the system tray icon"""
        if not PYSTRAY_AVAILABLE:
            return
        
        try:
            # Create a simple icon (you could load from file instead)
            from PIL import Image, ImageDraw
            
            # Create a simple microphone icon
            image = Image.new('RGB', (64, 64), color='white')
            draw = ImageDraw.Draw(image)
            
            # Draw a simple microphone shape
            draw.ellipse([20, 15, 44, 35], fill='black')  # Head
            draw.rectangle([30, 35, 34, 45], fill='black')  # Stand
            draw.rectangle([25, 45, 39, 50], fill='black')  # Base
            
            # Create menu
            menu = pystray.Menu(
                item('Show/Hide', self._toggle_window),
                item('Start/Stop Recording', self._toggle_recording_from_tray),
                pystray.Menu.SEPARATOR,
                item('Settings', self._show_settings_from_tray),
                item('History', self._show_history_from_tray),
                pystray.Menu.SEPARATOR,
                item('Exit', self._exit_from_tray)
            )
            
            self.tray_icon = pystray.Icon(
                name="SpeechToText",
                icon=image,
                title="Professional Speech-to-Text",
                menu=menu
            )
            
        except Exception as e:
            print(f"Error creating tray icon: {e}")
    
    def start_tray(self):
        """Start the system tray in a separate thread"""
        if self.tray_icon and PYSTRAY_AVAILABLE:
            threading.Thread(target=self.tray_icon.run, daemon=True).start()
    
    def stop_tray(self):
        """Stop the system tray"""
        if self.tray_icon and PYSTRAY_AVAILABLE:
            self.tray_icon.stop()
    
    def update_icon_recording(self, is_recording: bool):
        """Update the tray icon to show recording status"""
        if not self.tray_icon or not PYSTRAY_AVAILABLE:
            return
        
        try:
            from PIL import Image, ImageDraw
            
            # Create icon with recording indicator
            image = Image.new('RGB', (64, 64), color='white')
            draw = ImageDraw.Draw(image)
            
            # Draw microphone
            color = 'red' if is_recording else 'black'
            draw.ellipse([20, 15, 44, 35], fill=color)  # Head
            draw.rectangle([30, 35, 34, 45], fill=color)  # Stand
            draw.rectangle([25, 45, 39, 50], fill=color)  # Base
            
            # Add recording indicator
            if is_recording:
                draw.ellipse([45, 10, 55, 20], fill='red', outline='white', width=2)
            
            self.tray_icon.icon = image
            
        except Exception as e:
            print(f"Error updating tray icon: {e}")
    
    def minimize_to_tray(self):
        """Minimize window to system tray"""
        if PYSTRAY_AVAILABLE and self.app.config.system_tray:
            self.app.root.withdraw()  # Hide window
            self.is_visible = False
    
    def restore_from_tray(self):
        """Restore window from system tray"""
        self.app.root.deiconify()  # Show window
        self.app.root.lift()       # Bring to front
        self.is_visible = True
    
    def _toggle_window(self, icon=None, item=None):
        """Toggle window visibility from tray"""
        if self.is_visible:
            self.minimize_to_tray()
        else:
            self.restore_from_tray()
    
    def _toggle_recording_from_tray(self, icon=None, item=None):
        """Toggle recording from tray menu"""
        self.app._toggle_recording()
    
    def _show_settings_from_tray(self, icon=None, item=None):
        """Show settings from tray menu"""
        if not self.is_visible:
            self.restore_from_tray()
        self.app._open_settings()
    
    def _show_history_from_tray(self, icon=None, item=None):
        """Show history from tray menu"""
        if not self.is_visible:
            self.restore_from_tray()
        self.app._show_history()
    
    def _exit_from_tray(self, icon=None, item=None):
        """Exit application from tray menu"""
        self.app._on_window_close()


class TranscriptHistory:
    """Manages transcript history and export functionality"""

    def __init__(self, history_file="data/transcript_history.json"):
        self.history = []
        self.max_history = 100
        self.history_file = history_file
        self.gui_refresh_callback = None  # Callback to refresh GUI
        self._load_history()
    
    def add_transcript(self, text: str, timestamp: datetime = None, mode: str = ""):
        """Add a transcript to history"""
        if timestamp is None:
            timestamp = datetime.now()
            
        entry = {
            'text': text,
            'timestamp': timestamp,
            'mode': mode,
            'length': len(text)
        }
        
        self.history.insert(0, entry)  # Add to beginning
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[:self.max_history]
        
        # Save to file immediately
        self._save_history()
        
        # Refresh GUI if callback is set
        if self.gui_refresh_callback:
            self.gui_refresh_callback()
    
    def search_history(self, query: str) -> List[Dict]:
        """Search transcript history"""
        query_lower = query.lower()
        results = []
        
        for entry in self.history:
            if query_lower in entry['text'].lower():
                results.append(entry)
        
        return results
    
    def _load_history(self):
        """Load history from file"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Convert timestamp strings back to datetime objects
                for entry in data:
                    if isinstance(entry['timestamp'], str):
                        entry['timestamp'] = datetime.fromisoformat(entry['timestamp'])
                
                self.history = data
                print(f"Loaded {len(self.history)} transcript entries from history")
        except Exception as e:
            print(f"Error loading history: {e}")
            self.history = []
    
    def _save_history(self):
        """Save history to file"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Convert datetime objects to strings for JSON serialization
            data = []
            for entry in self.history:
                entry_copy = entry.copy()
                entry_copy['timestamp'] = entry['timestamp'].isoformat()
                data.append(entry_copy)
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def export_history(self, filename: str, format: str = "txt") -> bool:
        """Export transcript history to file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                if format == "txt":
                    for entry in self.history:
                        f.write(f"[{entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}] {entry['mode']}\n")
                        f.write(f"{entry['text']}\n\n")
                elif format == "json":
                    json.dump([{
                        'text': entry['text'],
                        'timestamp': entry['timestamp'].isoformat(),
                        'mode': entry['mode'],
                        'length': entry['length']
                    } for entry in self.history], f, indent=2)
            return True
        except Exception as e:
            print(f"Error exporting history: {e}")
            return False


# Continue with the main GUI application class...
class SpeechToTextGUI:
    """Main GUI application class"""
    
    def __init__(self):
        self.config = AppConfig.load_from_file()
        self.root = None
        self.audio_engine = None
        self.transcription_engine = None
        self.screenshot_engine = None
        self.hotkey_manager = HotkeyManager()
        self.transcript_history = TranscriptHistory()
        self.system_tray = None

        # GUI components
        self.main_frame = None
        self.control_frame = None
        self.transcription_display = None
        self.status_bar = None
        self.progress_var = None
        self.recording_indicator = None  # Recording overlay window
        self.status_var = None
        self.reprocess_button = None

        # Recording state
        self.is_recording = False
        self.current_transcript = ""
        self.recording_timer = None
        self.is_reprocessing = False

        # Last-run context for recovery/reprocessing
        self.last_recording_primary_path = None
        self.last_recording_backup_path = None
        self.last_recording_timestamp = None
        self.last_screenshot_image = None
        self.last_screenshot_path = None
        self.last_transcript_text = ""

        self._initialize_application()
    
    def _initialize_application(self):
        """Initialize the main application"""
        try:
            # Initialize engines
            self.audio_engine = AudioEngine(self.config)
            self.transcription_engine = TranscriptionEngine(self.config)
            self.screenshot_engine = ScreenshotEngine(self.config)
            self.correction_engine = CorrectionEngine(self.config)
            
            # Create GUI
            self._create_gui()
            
            # Setup hotkeys
            self._setup_hotkeys()
            
            # Setup system tray
            if self.config.system_tray:
                self.system_tray = SystemTrayManager(self)
                self.system_tray.start_tray()
            
            print("Application initialized successfully")
            print("📱 Recording Controls: Middle mouse button (scroll wheel click) or Ctrl+Q")
            print("⚙️  Configure recording options in Settings tab")
            
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {e}")
            sys.exit(1)
    
    def _create_gui(self):
        """Create the main GUI"""
        self.root = tk.Tk()
        self.root.title("Professional Speech-to-Text - Psynect Corp")
        self.root.geometry(f"{self.config.window_width}x{self.config.window_height}")
        
        # Apply theme
        self.theme = ThemeManager.apply_theme(self.root, self.config.theme)
        
        # Configure window
        self.root.minsize(800, 600)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Create main interface
        self._create_main_interface()
        
        # Create status bar
        self._create_status_bar()
        
        # Setup window events
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Center window
        self._center_window()
        
        # Set up history auto-refresh callback now that GUI is created
        self.transcript_history.gui_refresh_callback = self._refresh_history
    
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = Menu(self.root, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export History...", command=self._export_history)
        file_menu.add_command(label="Import Audio...", command=self._import_audio)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_window_close)
        
        # Edit menu
        edit_menu = Menu(menubar, tearoff=0, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Copy Transcript", command=self._copy_transcript)
        edit_menu.add_command(label="Clear Transcript", command=self._clear_transcript)
        edit_menu.add_separator()
        edit_menu.add_command(label="Settings...", command=self._open_settings)
        
        # Recording menu
        recording_menu = Menu(menubar, tearoff=0, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        menubar.add_cascade(label="Recording", menu=recording_menu)
        recording_menu.add_command(label="Start/Stop Recording", command=self._toggle_recording)
        recording_menu.add_separator()
        recording_menu.add_command(label="Audio Devices...", command=self._show_audio_devices)
        
        # View menu
        view_menu = Menu(menubar, tearoff=0, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Theme submenu
        theme_menu = Menu(view_menu, tearoff=0, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        view_menu.add_cascade(label="Theme", menu=theme_menu)
        theme_menu.add_command(label="Dark", command=lambda: self._change_theme("dark"))
        theme_menu.add_command(label="Light", command=lambda: self._change_theme("light"))
        theme_menu.add_command(label="Professional", command=lambda: self._change_theme("professional"))
        
        view_menu.add_separator()
        view_menu.add_command(label="History...", command=self._show_history)
        
        # Help menu
        help_menu = Menu(menubar, tearoff=0, bg=self.theme['menu_bg'], fg=self.theme['menu_fg'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_command(label="About...", command=self._show_about)
    
    def _create_main_interface(self):
        """Create the main application interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_container)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Recording tab
        self._create_recording_tab(notebook)
        
        # Settings tab
        self._create_settings_tab(notebook)
        
        # History tab
        self._create_history_tab(notebook)
    
    def _create_recording_tab(self, parent):
        """Create the main recording interface tab"""
        tab_frame = ttk.Frame(parent)
        parent.add(tab_frame, text="Recording")
        
        # Control panel at top
        control_panel = ttk.LabelFrame(tab_frame, text="Recording Controls", padding=10)
        control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Mode selection (matching original CLI structure)
        mode_frame = ttk.Frame(control_panel)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(mode_frame, text="Transcription Mode:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Initialize mode label from saved config to match original CLI behavior
        saved_mode = self.config.mode
        saved_backend = self.config.backend
        if saved_backend in [3, 4]:
            if saved_backend == 3:
                initial_mode_label = "Groq Whisper 3 Large with Screenshot Correction"
            else:
                initial_mode_label = "Groq Whisper 3 Large Transcription Only"
        elif saved_backend in [5, 6, 7]:
            if saved_backend == 5:
                initial_mode_label = "Deepgram Nova-3 (Highest Accuracy) with Correction"
            elif saved_backend == 6:
                initial_mode_label = "Deepgram Nova-3 Transcription Only"
            else:
                initial_mode_label = "Deepgram Flux (Fast) Transcription Only"
        else:
            initial_mode_label = {
                1: "High-Accuracy Mode (gpt-4o-transcribe + Screenshot Correction)",
                2: "Fast-Processing Mode (gpt-4o-mini models)",
                3: "Real-time Mode (gpt-4o-transcribe, no correction)",
                4: "Transcription-Only (High-Accuracy)",
                5: "Transcription-Only (Fast)"
            }.get(saved_mode, "High-Accuracy Mode (gpt-4o-transcribe + Screenshot Correction)")

        self.mode_var = tk.StringVar(value=initial_mode_label)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, state="readonly", width=50)
        mode_combo['values'] = [
            "High-Accuracy Mode (gpt-4o-transcribe + Screenshot Correction)",
            "Fast-Processing Mode (gpt-4o-mini models)", 
            "Real-time Mode (gpt-4o-transcribe, no correction)",
            "Transcription-Only (High-Accuracy)",
            "Transcription-Only (Fast)",
            "Groq Whisper 3 Large with Screenshot Correction",
            "Groq Whisper 3 Large Transcription Only",
            "Deepgram Nova-3 (Highest Accuracy) with Correction",
            "Deepgram Nova-3 Transcription Only",
            "Deepgram Flux (Fast) Transcription Only"
        ]
        mode_combo.pack(side=tk.LEFT, padx=(0, 10))
        mode_combo.bind('<<ComboboxSelected>>', self._on_mode_changed)
        
        # Recording button frame
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(fill=tk.X, pady=10)
        
        # Main record button
        self.record_button = ttk.Button(
            button_frame, 
            text="🎤 Start Recording", 
            command=self._toggle_recording,
            style="Record.TButton"
        )
        self.record_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Recording status
        self.recording_status = ttk.Label(button_frame, text="Ready to record")
        self.recording_status.pack(side=tk.LEFT, padx=(0, 10))
        
        # Recording timer
        self.timer_label = ttk.Label(button_frame, text="00:00")
        self.timer_label.pack(side=tk.RIGHT)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            control_panel, 
            variable=self.progress_var, 
            maximum=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Transcription display
        transcript_frame = ttk.LabelFrame(tab_frame, text="Live Transcription", padding=10)
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create scrolled text widget for transcription
        self.transcription_display = scrolledtext.ScrolledText(
            transcript_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            font=('Consolas', 11),
            bg=self.theme['text_bg'],
            fg=self.theme['text_fg'],
            insertbackground=self.theme['text_fg']
        )
        self.transcription_display.pack(fill=tk.BOTH, expand=True)
        
        # Action buttons frame
        action_frame = ttk.Frame(transcript_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(action_frame, text="Copy", command=self._copy_transcript).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Clear", command=self._clear_transcript).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(action_frame, text="Export...", command=self._export_transcript).pack(side=tk.LEFT, padx=(0, 5))
        self.reprocess_button = ttk.Button(
            action_frame,
            text="Reprocess Last Recording",
            command=self._reprocess_last_recording,
            state=tk.DISABLED
        )
        self.reprocess_button.pack(side=tk.LEFT, padx=(0, 5))

        # Auto-paste option
        self.auto_paste_var = tk.BooleanVar(value=self.config.auto_paste)
        ttk.Checkbutton(
            action_frame,
            text="Auto-paste after transcription",
            variable=self.auto_paste_var,
            command=self._on_auto_paste_changed
        ).pack(side=tk.RIGHT)

        # Ensure reprocess button reflects availability on startup
        self._update_reprocess_button_state()
    
    def _create_settings_tab(self, parent):
        """Create the settings configuration tab"""
        tab_frame = ttk.Frame(parent)
        parent.add(tab_frame, text="Settings")
        
        # Create scrollable frame for settings
        canvas = tk.Canvas(tab_frame, bg=self.theme['bg'])
        scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # API Configuration section
        api_frame = ttk.LabelFrame(scrollable_frame, text="API Configuration", padding=10)
        api_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Load API keys from environment if not in config (data/.env first)
        try:
            load_dotenv(dotenv_path=os.path.join('data', '.env'))
        except Exception:
            pass
        load_dotenv()
        openai_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        groq_key = self.config.groq_api_key or os.getenv("GROQ_API_KEY", "")
        anthropic_key = self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        deepgram_key = getattr(self.config, 'deepgram_api_key', '') or os.getenv("DEEPGRAM_API_KEY", "")
        
        # OpenAI API Key
        ttk.Label(api_frame, text="OpenAI API Key:").grid(row=0, column=0, sticky="w", pady=2)
        self.openai_key_var = tk.StringVar(value=openai_key)
        openai_entry = ttk.Entry(api_frame, textvariable=self.openai_key_var, show="*", width=50)
        openai_entry.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=2)
        
        # Groq API Key
        ttk.Label(api_frame, text="Groq API Key:").grid(row=1, column=0, sticky="w", pady=2)
        self.groq_key_var = tk.StringVar(value=groq_key)
        groq_entry = ttk.Entry(api_frame, textvariable=self.groq_key_var, show="*", width=50)
        groq_entry.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=2)
        
        # Anthropic API Key
        ttk.Label(api_frame, text="Anthropic API Key:").grid(row=2, column=0, sticky="w", pady=2)
        self.anthropic_key_var = tk.StringVar(value=anthropic_key)
        anthropic_entry = ttk.Entry(api_frame, textvariable=self.anthropic_key_var, show="*", width=50)
        anthropic_entry.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=2)
        
        # Deepgram API Key
        ttk.Label(api_frame, text="Deepgram API Key:").grid(row=3, column=0, sticky="w", pady=2)
        self.deepgram_key_var = tk.StringVar(value=deepgram_key)
        deepgram_entry = ttk.Entry(api_frame, textvariable=self.deepgram_key_var, show="*", width=50)
        deepgram_entry.grid(row=3, column=1, sticky="ew", padx=(5, 0), pady=2)

        # Google API Key (for Gemini)
        ttk.Label(api_frame, text="Google API Key (Gemini):").grid(row=4, column=0, sticky="w", pady=2)
        google_key = getattr(self.config, 'google_api_key', '') or os.getenv("GOOGLE_API_KEY", "")
        self.google_key_var = tk.StringVar(value=google_key)
        google_entry = ttk.Entry(api_frame, textvariable=self.google_key_var, show="*", width=50)
        google_entry.grid(row=4, column=1, sticky="ew", padx=(5, 0), pady=2)

        api_frame.columnconfigure(1, weight=1)

        # Test API connections button
        ttk.Button(api_frame, text="Test Connections", command=self._test_api_connections).grid(
            row=5, column=1, sticky="e", pady=10
        )
        
        # Recording Settings section
        recording_frame = ttk.LabelFrame(scrollable_frame, text="Recording Settings", padding=10)
        recording_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Hotkey configuration
        ttk.Label(recording_frame, text="Recording Hotkey:").grid(row=0, column=0, sticky="w", pady=2)
        self.hotkey_var = tk.StringVar(value=self.config.hotkey)
        # Use tk.Entry to ensure foreground/background colors are visible in dark theme
        hotkey_entry = tk.Entry(recording_frame, textvariable=self.hotkey_var, width=22,
                                bg=self.theme['entry_bg'], fg=self.theme['entry_fg'],
                                insertbackground=self.theme.get('entry_insert_bg', self.theme['entry_fg']),
                                relief='solid', bd=1)
        hotkey_entry.grid(row=0, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # Mouse button setting
        ttk.Label(recording_frame, text="Enable Middle Mouse Button:").grid(row=1, column=0, sticky="w", pady=2)
        self.mouse_button_var = tk.BooleanVar(value=self.config.enable_mouse_button)
        mouse_check = ttk.Checkbutton(
            recording_frame,
            variable=self.mouse_button_var,
            command=self._on_mouse_button_changed
        )
        mouse_check.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # Add helpful note about mouse button
        mouse_note = tk.Label(recording_frame, 
                             text="Click scroll wheel to record (enabled by default)",
                             font=("Arial", 8), fg="gray")
        mouse_note.grid(row=1, column=2, sticky="w", padx=(10, 0), pady=2)
        
        # Max recording duration
        ttk.Label(recording_frame, text="Max Recording Duration (seconds):").grid(row=2, column=0, sticky="w", pady=2)
        self.max_duration_var = tk.IntVar(value=self.config.max_recording_duration)
        ttk.Spinbox(
            recording_frame, 
            from_=10, 
            to=3600, 
            textvariable=self.max_duration_var,
            width=10
        ).grid(row=2, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # Audio quality settings
        ttk.Label(recording_frame, text="Sample Rate (Hz):").grid(row=3, column=0, sticky="w", pady=2)
        self.sample_rate_var = tk.IntVar(value=self.config.rate)
        rate_combo = ttk.Combobox(recording_frame, textvariable=self.sample_rate_var, width=15, state="readonly")
        rate_combo['values'] = [16000, 22050, 24000, 44100, 48000]
        rate_combo.grid(row=3, column=1, sticky="w", padx=(5, 0), pady=2)

        # Correction Model selection
        ttk.Label(recording_frame, text="Correction Model:").grid(row=4, column=0, sticky="w", pady=2)
        self._correction_model_map = {
            "Gemini 3 Flash": "gemini",
            "GPT-4.1 (OpenAI)": "openai",
        }
        self._correction_model_reverse = {v: k for k, v in self._correction_model_map.items()}
        current_value = getattr(self.config, 'correction_model', 'gemini')
        display_value = self._correction_model_reverse.get(current_value, "Gemini 3 Flash")
        self.correction_model_var = tk.StringVar(value=display_value)
        correction_combo = ttk.Combobox(recording_frame, textvariable=self.correction_model_var, width=22, state="readonly")
        correction_combo['values'] = list(self._correction_model_map.keys())
        correction_combo.grid(row=4, column=1, sticky="w", padx=(5, 0), pady=2)

        # Add helpful note about correction model
        correction_note = tk.Label(recording_frame,
                             text="Gemini 3 Flash (cheaper) or GPT-4.1",
                             font=("Arial", 8), fg="gray")
        correction_note.grid(row=4, column=2, sticky="w", padx=(10, 0), pady=2)

        # UI Settings section
        ui_frame = ttk.LabelFrame(scrollable_frame, text="User Interface", padding=10)
        ui_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Theme selection
        ttk.Label(ui_frame, text="Theme:").grid(row=0, column=0, sticky="w", pady=2)
        self.theme_var = tk.StringVar(value=self.config.theme)
        theme_combo = ttk.Combobox(ui_frame, textvariable=self.theme_var, width=15, state="readonly")
        theme_combo['values'] = ["dark", "light", "professional"]
        theme_combo.grid(row=0, column=1, sticky="w", padx=(5, 0), pady=2)
        theme_combo.bind('<<ComboboxSelected>>', lambda e: self._change_theme(self.theme_var.get()))
        
        # System tray options
        self.auto_minimize_var = tk.BooleanVar(value=self.config.auto_minimize)
        ttk.Checkbutton(
            ui_frame, 
            text="Minimize to system tray when recording", 
            variable=self.auto_minimize_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        
        self.system_tray_var = tk.BooleanVar(value=self.config.system_tray)
        ttk.Checkbutton(
            ui_frame, 
            text="Enable system tray integration", 
            variable=self.system_tray_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=2)
        
        # Save settings button
        settings_button_frame = ttk.Frame(scrollable_frame)
        settings_button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            settings_button_frame, 
            text="Save Settings", 
            command=self._save_settings
        ).pack(side=tk.RIGHT)
        
        ttk.Button(
            settings_button_frame, 
            text="Reset to Defaults", 
            command=self._reset_settings
        ).pack(side=tk.RIGHT, padx=(0, 10))
    
    def _create_history_tab(self, parent):
        """Create the transcript history tab"""
        tab_frame = ttk.Frame(parent)
        parent.add(tab_frame, text="History")
        
        # Search frame
        search_frame = ttk.Frame(tab_frame)
        search_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(search_frame, text="Search:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=tk.LEFT, padx=(0, 5))
        search_entry.bind('<Return>', self._search_history)
        
        ttk.Button(search_frame, text="Search", command=self._search_history).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(search_frame, text="Clear", command=self._clear_search).pack(side=tk.LEFT)
        
        # History list frame
        history_frame = ttk.Frame(tab_frame)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for history
        columns = ('timestamp', 'mode', 'length', 'preview')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        self.history_tree.heading('timestamp', text='Timestamp')
        self.history_tree.heading('mode', text='Mode')
        self.history_tree.heading('length', text='Length')
        self.history_tree.heading('preview', text='Preview')
        
        self.history_tree.column('timestamp', width=150)
        self.history_tree.column('mode', width=100)
        self.history_tree.column('length', width=80)
        self.history_tree.column('preview', width=400)
        
        # Add scrollbar for history
        history_scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_tree.pack(side="left", fill="both", expand=True)
        history_scrollbar.pack(side="right", fill="y")
        
        # Bind double-click to view full transcript
        self.history_tree.bind('<Double-1>', self._view_transcript)
        
        # History action buttons
        history_actions = ttk.Frame(tab_frame)
        history_actions.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(history_actions, text="View", command=self._view_selected_transcript).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(history_actions, text="Copy", command=self._copy_selected_transcript).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(history_actions, text="Delete", command=self._delete_selected_transcript).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(history_actions, text="Export All...", command=self._export_all_history).pack(side=tk.RIGHT)
        ttk.Button(history_actions, text="Clear All", command=self._clear_all_history).pack(side=tk.RIGHT, padx=(0, 10))
        
        # Load initial history
        self._refresh_history()
    
    def _create_status_bar(self):
        """Create the application status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Status text
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Service status indicators
        services_frame = ttk.Frame(self.status_frame)
        services_frame.pack(side=tk.RIGHT, padx=5)
        
        self.service_labels = {}
        services = self.transcription_engine.get_available_services()
        
        for service, available in services.items():
            color = "green" if available else "red"
            label = ttk.Label(services_frame, text=f"{service.upper()}: {'✓' if available else '✗'}")
            label.pack(side=tk.RIGHT, padx=5)
            self.service_labels[service] = label
    
    def _center_window(self):
        """Center the main window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def _setup_hotkeys(self):
        """Setup global hotkeys"""
        self.hotkey_manager.register_hotkey(self.config.hotkey, self._toggle_recording)
        if MOUSE_AVAILABLE:
            self.hotkey_manager.register_mouse_button("middle", self._toggle_recording)
    
    # Event handlers and methods continue...
    def _toggle_recording(self):
        """Toggle recording state"""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            return

        if self.is_reprocessing:
            messagebox.showinfo("Recording", "Please wait for reprocessing to finish before starting a new recording.")
            return

        if not self.audio_engine.start_recording():
            error_message = self.audio_engine.last_error_message or "Failed to start recording"
            self._recording_error(error_message)
            if self.audio_engine:
                self.audio_engine.recover_after_error()
            self._update_reprocess_button_state()
            return

        # Update UI after successful start
        self.is_recording = True
        self.record_button.config(text="🛑 Stop Recording")
        self.recording_status.config(text="Recording...")
        self.status_var.set("Recording in progress...")

        # Show recording indicator overlay
        self._show_recording_indicator()

        # Clear previous transcript
        self.transcription_display.delete(1.0, tk.END)

        # Start timer and progress indicator
        self._start_recording_timer()
        self.progress_var.set(0)
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        # Auto-minimize if enabled
        if self.config.auto_minimize:
            if self.system_tray and PYSTRAY_AVAILABLE:
                self.system_tray.minimize_to_tray()
            else:
                self.root.iconify()

        # Update tray icon
        if self.system_tray:
            self.system_tray.update_icon_recording(True)

    def _stop_recording(self):
        """Stop audio recording and process"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        
        # Update UI
        self.record_button.config(text="🎤 Start Recording")
        self.recording_status.config(text="Processing...")
        self.status_var.set("Processing audio...")
        
        # Keep cursor as "watch" during processing
        
        # Stop recording timer
        if self.recording_timer:
            self.root.after_cancel(self.recording_timer)
            self.recording_timer = None
        
        # Stop progress bar
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        
        # Update tray icon
        if self.system_tray:
            self.system_tray.update_icon_recording(False)
        
        # Stop audio recording
        success, duration = self.audio_engine.stop_recording()
        
        if success:
            # Save recording and process
            threading.Thread(target=self._process_recording, daemon=True).start()
        else:
            self._recording_error("Failed to stop recording")
    
    def _process_recording(self):
        """Process the recorded audio in background thread"""
        try:
            # PRIORITY #1: Save the audio recording FIRST - this is critical!
            print("🔒 SAVING AUDIO RECORDING - TOP PRIORITY...")
            
            # Set up file paths using ABSOLUTE paths (not relative)
            # This fixes WinError 433 when working directory becomes unavailable (e.g., Google Drive after sleep)
            try:
                recordings_dir = os.path.join(SCRIPT_DIR, "recordings")
                os.makedirs(recordings_dir, exist_ok=True)
                print(f"📁 Using primary recordings directory: {recordings_dir}")
            except OSError as path_error:
                # Primary path failed (e.g., Google Drive unavailable) - fall back to Documents folder
                print(f"⚠️  Primary recordings directory unavailable: {path_error}")
                recordings_dir = self.audio_engine.get_fallback_documents_dir()
                print(f"📁 Falling back to: {recordings_dir}")
            
            temp_file = os.path.join(recordings_dir, "latest_recording.wav")
            
            # Use robust saving with automatic fallback to Documents folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(recordings_dir, f"recording_{timestamp}.wav")
            
            success, saved_path, backup_path = self.audio_engine.safe_save_recording(temp_file, backup_file)
            
            if not success:
                self.root.after(0, lambda: self._recording_error("🚨 CRITICAL ERROR: Could not save audio recording anywhere!"))
                return
            
            # Log what was saved
            print(f"✅ PRIMARY: Audio saved to: {saved_path}")
            if backup_path:
                print(f"✅ BACKUP: Timestamped copy: {backup_path}")
            
            # Get file size for validation
            file_size = os.path.getsize(saved_path)
            file_size_mb = file_size / (1024 * 1024)
            print(f"📊 Audio file size: {file_size_mb:.2f} MB")
            
            # Validate audio file has actual content
            if file_size < 1000:  # Less than 1KB indicates a problem
                print(f"⚠️  WARNING: Audio file suspiciously small ({file_size} bytes) - may indicate recording issue")
                print("⚠️  Audio recording saved but may be corrupted. Processing will continue.")
            else:
                print(f"✅ Audio file size looks good: {file_size_mb:.2f} MB")

            print("🔍 DEBUG: Step 1 - Persisting recording details...")
            sys.stdout.flush()  # Ensure output is visible before potential crash
            # Persist details for recovery/reprocessing
            self.last_recording_primary_path = saved_path
            self.last_recording_backup_path = backup_path
            self.last_recording_timestamp = datetime.now()
            print("🔍 DEBUG: Step 2 - Scheduling reprocess button update...")
            sys.stdout.flush()
            self.root.after(0, self._update_reprocess_button_state)

            # NOTE: Do NOT clear recorded_frames here during processing!
            # This causes race conditions with the recording thread.
            # Frames will be cleared when the next recording starts or in cleanup.

            print("🔍 DEBUG: Step 3 - Updating progress bar...")
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(25))

            # Capture screenshot if correction is enabled
            screenshot = None
            screenshot_path = None
            if self.config.use_correction:
                try:
                    screenshot, screenshot_path = self.screenshot_engine.capture_screenshot()
                    if screenshot_path:
                        print(f"📸 Screenshot saved to: {screenshot_path}")
                    self.root.after(0, lambda: self.progress_var.set(40))
                except Exception as capture_error:
                    print(f"Screenshot capture failed: {capture_error}")
                    screenshot = None
                    screenshot_path = None

            # Store last screenshot for potential reprocessing
            if screenshot is not None:
                try:
                    self.last_screenshot_image = screenshot.copy()
                except Exception:
                    self.last_screenshot_image = screenshot
            else:
                self.last_screenshot_image = None
            self.last_screenshot_path = screenshot_path

            print("🔍 DEBUG: Step 5 - Starting transcription pipeline...")
            print(f"🔍 DEBUG: Audio path: {saved_path}")
            print(f"🔍 DEBUG: Screenshot available: {screenshot is not None}")
            print(f"🔍 DEBUG: Backend setting: {self.config.backend}")
            
            # Run the transcription/correction pipeline on the saved audio
            # Wrap in additional try/except to catch any segfaults or C-level crashes
            try:
                self._run_transcription_pipeline(saved_path, screenshot, triggered_by_reprocess=False)
                print("🔍 DEBUG: Step 6 - Transcription pipeline returned successfully")
            except Exception as pipeline_error:
                print(f"❌ DEBUG: Pipeline crashed with: {pipeline_error}")
                traceback.print_exc()
                raise  # Re-raise to be caught by outer handler

        except Exception as e:
            # Catch ALL exceptions and display them properly without crashing
            error_msg = f"Processing error: {e}"
            print(f"❌ {error_msg}")
            traceback.print_exc()
            # Use lambda with default argument to avoid closure issues
            self.root.after(0, lambda msg=error_msg: self._recording_error(msg))



    def _run_transcription_pipeline(self, audio_path: str, screenshot: Optional[Image.Image], triggered_by_reprocess: bool = False):
        """Run transcription (and optional correction) for the provided audio file."""
        print("🔍 DEBUG: Entered _run_transcription_pipeline")

        def progress_callback(message, progress):
            self.root.after(0, lambda: self.status_var.set(message))
            self.root.after(0, lambda: self.progress_var.set(progress))

        if triggered_by_reprocess:
            self.root.after(0, self._prepare_progress_bar_for_reprocess)

        print("🔍 DEBUG: About to call transcribe_file...")
        try:
            success, transcript = self.transcription_engine.transcribe_file(audio_path, progress_callback)
            print(f"🔍 DEBUG: transcribe_file returned. Success={success}, Length={len(transcript) if transcript else 0}")
        except Exception as e:
            print(f"❌ DEBUG: transcribe_file crashed: {e}")
            traceback.print_exc()
            success = False
            transcript = f"Transcription error: {e}"

        if success:
            print("🔍 DEBUG: Transcription successful, processing result...")
            final_transcript = transcript
            if self.config.use_correction and screenshot:
                print("🔍 DEBUG: Applying correction with screenshot...")
                self.root.after(0, lambda: self.status_var.set("Applying context correction..."))
                self.root.after(0, lambda: self.progress_var.set(85))
                try:
                    final_transcript = self._apply_correction(transcript, screenshot)
                    print(f"🔍 DEBUG: Correction applied, final length: {len(final_transcript)}")
                except Exception as corr_err:
                    print(f"❌ DEBUG: Correction crashed: {corr_err}")
                    traceback.print_exc()
                    final_transcript = transcript  # Use original on failure

            def on_success(result_text=final_transcript):
                self._transcription_complete(result_text)
                if triggered_by_reprocess:
                    self.recording_status.config(text="Reprocess complete")
                    self.status_var.set(f"Reprocessed {len(result_text)} characters")
                mode_text = self.mode_var.get()
                history_mode = f"{mode_text} (Reprocessed)" if triggered_by_reprocess else mode_text
                self.transcript_history.add_transcript(result_text, mode=history_mode)
                self.is_reprocessing = False
                self._update_reprocess_button_state()

            self.root.after(0, on_success)
        else:
            error_message = transcript

            def on_failure(message=error_message):
                if triggered_by_reprocess:
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate')
                    self.progress_var.set(0)
                    self.recording_status.config(text="Reprocess failed")
                    self.status_var.set(f"Reprocess failed: {message}")
                    messagebox.showerror("Reprocess Failed", message)
                    self.is_reprocessing = False
                    self._update_reprocess_button_state()
                else:
                    self._recording_error(message)

            self.root.after(0, on_failure)

    def _apply_correction(self, transcript: str, screenshot: Image.Image) -> str:
        """Apply screenshot-based correction to transcript"""
        try:
            return self.correction_engine.apply_correction(transcript, screenshot)
        except Exception as e:
            print(f"Error during correction: {e}")
            return transcript  # Return original transcript if correction fails
    
    def _transcription_complete(self, transcript: str):
        """Handle completed transcription"""
        self.last_transcript_text = transcript

        # Update display
        self.transcription_display.delete(1.0, tk.END)
        self.transcription_display.insert(1.0, transcript)

        # Update status
        self.recording_status.config(text="Transcription complete")
        self.status_var.set(f"Transcribed {len(transcript)} characters")
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.progress_var.set(100)

        # Copy to clipboard with retry logic
        clipboard_success = False
        for attempt in range(3):  # Try up to 3 times
            try:
                pyperclip.copy(transcript)
                clipboard_success = True
                break
            except Exception as e:
                if attempt < 2:  # Wait and retry
                    time.sleep(0.1)
                    continue
                else:
                    print(f"Clipboard error after {attempt + 1} attempts: {e}")
                    self.status_var.set("Clipboard access failed - transcript saved to history")
        
        # Auto-paste if enabled and clipboard succeeded
        if clipboard_success and self.auto_paste_var.get():
            self.root.after(500, lambda: pyautogui.hotkey('ctrl', 'v'))
        
        # Reset progress after delay and hide recording indicator
        self.root.after(2000, lambda: self.progress_var.set(0))
        
        # Hide recording indicator overlay
        self._hide_recording_indicator()
        
        # Clear recorded data now that processing is complete
        if self.audio_engine and hasattr(self.audio_engine, 'recorded_data'):
            with self.audio_engine._lock:
                self.audio_engine.recorded_data = []
            if hasattr(self, 'status_var'):
                self.root.after(3000, lambda: self.status_var.set("Ready for next recording"))
        
        # Don't automatically restore window - let user manually restore when needed
        # This prevents focus stealing when auto-paste is enabled
        
        # Update system tray tooltip if minimized to indicate completion
        if self.system_tray and not self.system_tray.is_visible:
            try:
                if hasattr(self.system_tray, 'icon') and self.system_tray.icon:
                    self.system_tray.icon.title = f"Transcription Complete ({len(transcript)} chars) - Click to view"
            except Exception:
                pass

        self._update_reprocess_button_state()
    
    def _recording_error(self, message: str):
        """Handle recording errors"""
        self.is_recording = False
        self.record_button.config(text="🎤 Start Recording")
        self.recording_status.config(text="Error")
        self.status_var.set(f"Error: {message}")
        self.progress_var.set(0)
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')

        # Hide recording indicator overlay
        self._hide_recording_indicator()

        if self.recording_timer:
            self.root.after_cancel(self.recording_timer)
            self.recording_timer = None

        messagebox.showerror("Recording Error", message)

        if self.audio_engine:
            self.audio_engine.recover_after_error()

        self._update_reprocess_button_state()

    def _get_available_last_recording_path(self) -> Optional[str]:
        """Return the path to the most recent saved recording if it exists."""
        for path in [self.last_recording_primary_path, self.last_recording_backup_path]:
            if path and os.path.exists(path):
                return path
        return None

    def _update_reprocess_button_state(self):
        """Enable or disable the reprocess button based on saved recordings."""
        if not self.reprocess_button:
            return

        has_recording = self._get_available_last_recording_path() is not None
        if self.is_reprocessing or not has_recording:
            self.reprocess_button.config(state=tk.DISABLED)
        else:
            self.reprocess_button.config(state=tk.NORMAL)

    def _reprocess_last_recording(self):
        """Re-run transcription/correction on the last saved recording."""
        if self.is_recording:
            messagebox.showinfo("Reprocess", "Stop the active recording before reprocessing.")
            return

        if self.is_reprocessing:
            return

        audio_path = self._get_available_last_recording_path()
        if not audio_path:
            messagebox.showwarning("Reprocess", "No saved recording available to reprocess.")
            self._update_reprocess_button_state()
            return

        self.is_reprocessing = True
        self.reprocess_button.config(state=tk.DISABLED)
        self.status_var.set("Reprocessing last recording...")
        self.recording_status.config(text="Reprocessing...")
        self.progress_var.set(0)
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()

        threading.Thread(target=self._reprocess_worker, args=(audio_path,), daemon=True).start()

    def _handle_reprocess_exception(self, message: str):
        """Handle unexpected errors during reprocessing."""
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.progress_var.set(0)
        self.status_var.set(message)
        self.recording_status.config(text="Reprocess failed")
        messagebox.showerror("Reprocess Error", message)
        self.is_reprocessing = False
        self._update_reprocess_button_state()

    def _prepare_progress_bar_for_reprocess(self):
        """Switch the progress bar back to determinate mode for reprocessing."""
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')

    def _reprocess_worker(self, audio_path: str):
        """Background worker to reprocess the last saved recording."""
        try:
            screenshot = self.last_screenshot_image
            if self.config.use_correction and screenshot is None:
                try:
                    screenshot, screenshot_path = self.screenshot_engine.capture_screenshot()
                    if screenshot_path:
                        print(f"📸 Screenshot saved to: {screenshot_path}")
                    if screenshot is not None:
                        try:
                            self.last_screenshot_image = screenshot.copy()
                        except Exception:
                            self.last_screenshot_image = screenshot
                        self.last_screenshot_path = screenshot_path
                except Exception as capture_error:
                    print(f"Screenshot capture failed during reprocess: {capture_error}")
                    screenshot = None

            self._run_transcription_pipeline(audio_path, screenshot, triggered_by_reprocess=True)
        except Exception as e:
            error_message = f"Reprocess error: {e}"
            self.root.after(0, lambda msg=error_message: self._handle_reprocess_exception(msg))
    
    def _start_recording_timer(self):
        """Start the recording duration timer"""
        start_time = time.time()
        
        def update_timer():
            if self.is_recording:
                elapsed = int(time.time() - start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                self.timer_label.config(text=f"{minutes:02d}:{seconds:02d}")
                
                # Check max duration
                if elapsed >= self.config.max_recording_duration:
                    self._stop_recording()
                    messagebox.showwarning(
                        "Recording Limit", 
                        f"Recording stopped after reaching {self.config.max_recording_duration} second limit"
                    )
                    return
                
                self.recording_timer = self.root.after(1000, update_timer)
        
        update_timer()
    
    # Additional utility methods...
    def _copy_transcript(self):
        """Copy current transcript to clipboard"""
        transcript = self.transcription_display.get(1.0, tk.END).strip()
        if transcript:
            try:
                pyperclip.copy(transcript)
                self.status_var.set("Transcript copied to clipboard")
            except Exception as e:
                self.status_var.set(f"Clipboard error: {e}")
    
    def _clear_transcript(self):
        """Clear the transcription display"""
        self.transcription_display.delete(1.0, tk.END)
        self.status_var.set("Transcript cleared")
    
    def _show_recording_indicator(self):
        """Show a persistent recording indicator overlay"""
        if self.recording_indicator:
            return  # Already showing
            
        # Create a small overlay window
        self.recording_indicator = tk.Toplevel()
        self.recording_indicator.title("Recording")
        self.recording_indicator.geometry("200x60")
        
        # Make it stay on top
        self.recording_indicator.wm_attributes("-topmost", True)
        self.recording_indicator.wm_attributes("-toolwindow", True)  # No taskbar entry
        
        # Position it in the bottom-right corner
        screen_width = self.recording_indicator.winfo_screenwidth()
        screen_height = self.recording_indicator.winfo_screenheight()
        x = screen_width - 220  # 200 width + 20 margin
        y = screen_height - 100  # 60 height + 40 margin
        self.recording_indicator.geometry(f"200x60+{x}+{y}")
        
        # Style the window
        self.recording_indicator.configure(bg='#ff4444')
        self.recording_indicator.overrideredirect(True)  # Remove window decorations
        
        # Add content
        label = tk.Label(
            self.recording_indicator,
            text="🔴 RECORDING...",
            font=("Arial", 12, "bold"),
            fg="white",
            bg="#ff4444"
        )
        label.pack(expand=True)
        
        # Add pulsing animation
        self._animate_recording_indicator()
    
    def _hide_recording_indicator(self):
        """Hide the recording indicator overlay"""
        if self.recording_indicator:
            self.recording_indicator.destroy()
            self.recording_indicator = None
    
    def _animate_recording_indicator(self):
        """Animate the recording indicator with pulsing effect"""
        if not self.recording_indicator:
            return
            
        try:
            # Toggle between bright and dim red
            current_bg = self.recording_indicator.cget("bg")
            new_bg = "#ff6666" if current_bg == "#ff4444" else "#ff4444"
            
            self.recording_indicator.configure(bg=new_bg)
            for widget in self.recording_indicator.winfo_children():
                widget.configure(bg=new_bg)
                
            # Schedule next animation frame
            self.root.after(800, self._animate_recording_indicator)
        except tk.TclError:
            # Window was destroyed
            pass
    
    def _export_transcript(self):
        """Export current transcript to file"""
        transcript = self.transcription_display.get(1.0, tk.END).strip()
        if not transcript:
            messagebox.showwarning("Export", "No transcript to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export Transcript",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                self.status_var.set(f"Transcript exported to {filename}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export transcript: {e}")
    
    def _change_theme(self, theme_name: str):
        """Change application theme"""
        self.config.theme = theme_name
        self.theme = ThemeManager.apply_theme(self.root, theme_name)
        
        # Update specific widgets that need manual theme updates
        if hasattr(self, 'transcription_display'):
            self.transcription_display.config(
                bg=self.theme['text_bg'],
                fg=self.theme['text_fg'],
                insertbackground=self.theme['text_fg'],
                selectbackground=self.theme['select_bg'],
                selectforeground=self.theme['select_fg']
            )
        
        # Update history tree if it exists
        if hasattr(self, 'history_tree'):
            self.history_tree.configure(
                background=self.theme['text_bg'],
                foreground=self.theme['text_fg']
            )
        
        self.status_var.set(f"Theme changed to {theme_name}")
    
    def _save_settings(self):
        """Save current settings to configuration"""
        # Update config from UI
        self.config.openai_api_key = self.openai_key_var.get()
        self.config.groq_api_key = self.groq_key_var.get()
        self.config.anthropic_api_key = self.anthropic_key_var.get()
        if hasattr(self, 'deepgram_key_var'):
            self.config.deepgram_api_key = self.deepgram_key_var.get()
        self.config.hotkey = self.hotkey_var.get()
        self.config.max_recording_duration = self.max_duration_var.get()
        self.config.rate = self.sample_rate_var.get()
        self.config.theme = self.theme_var.get()
        self.config.auto_minimize = self.auto_minimize_var.get()
        self.config.system_tray = self.system_tray_var.get()
        self.config.auto_paste = self.auto_paste_var.get()
        if hasattr(self, 'google_key_var'):
            self.config.google_api_key = self.google_key_var.get()
        if hasattr(self, 'correction_model_var'):
            display = self.correction_model_var.get()
            self.config.correction_model = self._correction_model_map.get(display, 'gemini')

        # Save to file
        self.config.save_to_file()

        # Reinitialize engines with new settings
        self.transcription_engine = TranscriptionEngine(self.config)
        self.correction_engine = CorrectionEngine(self.config)
        
        # Update hotkeys
        self.hotkey_manager.unregister_all()
        self._setup_hotkeys()
        
        self.status_var.set("Settings saved successfully")
        messagebox.showinfo("Settings", "Settings saved successfully")
    
    def _reset_settings(self):
        """Reset all settings to defaults"""
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to defaults?"):
            self.config = AppConfig()
            
            # Update UI with default values
            self.openai_key_var.set(self.config.openai_api_key)
            self.groq_key_var.set(self.config.groq_api_key)
            self.anthropic_key_var.set(self.config.anthropic_api_key)
            if hasattr(self, 'deepgram_key_var'):
                self.deepgram_key_var.set(getattr(self.config, 'deepgram_api_key', ''))
            if hasattr(self, 'google_key_var'):
                self.google_key_var.set(getattr(self.config, 'google_api_key', ''))
            if hasattr(self, 'correction_model_var'):
                raw = getattr(self.config, 'correction_model', 'gemini')
                self.correction_model_var.set(self._correction_model_reverse.get(raw, 'Gemini 3 Flash'))
            self.hotkey_var.set(self.config.hotkey)
            self.max_duration_var.set(self.config.max_recording_duration)
            self.sample_rate_var.set(self.config.rate)
            self.theme_var.set(self.config.theme)
            self.auto_minimize_var.set(self.config.auto_minimize)
            self.system_tray_var.set(self.config.system_tray)
            self.auto_paste_var.set(self.config.auto_paste)
            
            self.status_var.set("Settings reset to defaults")
    
    def _test_api_connections(self):
        """Test API connections"""
        results = []
        
        # Test OpenAI
        if self.openai_key_var.get():
            try:
                client = OpenAI(api_key=self.openai_key_var.get())
                # Simple test - list models (this is a lightweight operation)
                models = client.models.list()
                results.append("✓ OpenAI: Connected")
            except Exception as e:
                results.append(f"✗ OpenAI: {str(e)[:50]}...")
        else:
            results.append("⚠ OpenAI: No API key")
        
        # Test Groq
        if self.groq_key_var.get():
            try:
                client = Groq(api_key=self.groq_key_var.get())
                # Test with a minimal request
                results.append("✓ Groq: Connected")
            except Exception as e:
                results.append(f"✗ Groq: {str(e)[:50]}...")
        else:
            results.append("⚠ Groq: No API key")
        
        # Test Anthropic
        if self.anthropic_key_var.get():
            try:
                client = Anthropic(api_key=self.anthropic_key_var.get())
                results.append("✓ Anthropic: Connected")
            except Exception as e:
                results.append(f"✗ Anthropic: {str(e)[:50]}...")
        else:
            results.append("⚠ Anthropic: No API key")
            
        # Test Deepgram
        deepgram_key = getattr(self, 'deepgram_key_var', tk.StringVar()).get()
        if deepgram_key:
            try:
                headers = {"Authorization": f"Token {deepgram_key}"}
                response = requests.get("https://api.deepgram.com/v1/projects", headers=headers)
                response.raise_for_status()
                results.append("✓ Deepgram: Connected")
            except Exception as e:
                results.append(f"✗ Deepgram: {str(e)[:50]}...")
        else:
            results.append("⚠ Deepgram: No API key")

        # Test Google Gemini
        google_key = getattr(self, 'google_key_var', tk.StringVar()).get()
        if google_key:
            try:
                if GEMINI_AVAILABLE:
                    test_client = genai.Client(api_key=google_key)
                    # Actually test the connection
                    test_client.models.list()
                    results.append("✓ Google Gemini: Connected")
                else:
                    results.append("✗ Google Gemini: SDK not installed (pip install google-genai)")
            except Exception as e:
                results.append(f"✗ Google Gemini: {str(e)[:50]}...")
        else:
            results.append("⚠ Google Gemini: No API key")

        messagebox.showinfo("API Connection Test", "\n".join(results))
    
    def _on_mode_changed(self, event=None):
        """Handle mode selection change - matches original CLI logic"""
        mode = self.mode_var.get()
        
        # Map GUI modes to original CLI modes and backends
        if "High-Accuracy Mode" in mode:
            self.config.mode = 1
            self.config.backend = 1  # GPT-4o
            self.config.use_correction = True
        elif "Fast-Processing Mode" in mode:
            self.config.mode = 2
            self.config.backend = 1  # GPT-4o-mini
            self.config.use_correction = True
        elif "Real-time Mode" in mode:
            self.config.mode = 3
            self.config.backend = 1  # GPT-4o
            self.config.use_correction = False
        elif "Transcription-Only (High-Accuracy)" in mode:
            self.config.mode = 4
            self.config.backend = 1  # GPT-4o
            self.config.use_correction = False
        elif "Transcription-Only (Fast)" in mode:
            self.config.mode = 5
            self.config.backend = 1  # GPT-4o-mini
            self.config.use_correction = False
        elif "Groq Whisper 3 Large with Screenshot Correction" in mode:
            self.config.mode = 1
            self.config.backend = 3  # Groq
            self.config.use_correction = True
        elif "Groq Whisper 3 Large Transcription Only" in mode:
            self.config.mode = 4
            self.config.backend = 4  # Groq only
            self.config.use_correction = False
        elif "Deepgram Nova-3 (Highest Accuracy) with Correction" in mode:
            self.config.mode = 1
            self.config.backend = 5
            self.config.use_correction = True
        elif "Deepgram Nova-3 Transcription Only" in mode:
            self.config.mode = 4
            self.config.backend = 6
            self.config.use_correction = False
        elif "Deepgram Flux (Fast) Transcription Only" in mode:
            self.config.mode = 5
            self.config.backend = 7
            self.config.use_correction = False
        
        # Save the updated config to persist the mode selection
        self.config.save_to_file()
    
    def _on_auto_paste_changed(self):
        """Handle auto-paste option change"""
        self.config.auto_paste = self.auto_paste_var.get()
        # Save the updated config
        self.config.save_to_file()
        
    def _on_mouse_button_changed(self):
        """Handle mouse button enable/disable change"""
        self.config.enable_mouse_button = self.mouse_button_var.get()
        # Save the updated config
        self.config.save_to_file()
        # Re-setup hotkeys to apply the change
        self._setup_hotkeys()
    
    def _setup_hotkeys(self):
        """Setup global hotkeys for recording"""
        try:
            # Clear any existing hotkeys first
            self.hotkey_manager.unregister_all()
            
            # Register keyboard hotkey
            if self.config.hotkey and self.config.hotkey.strip():
                success = self.hotkey_manager.register_hotkey(self.config.hotkey, self._toggle_recording)
                if success:
                    print(f"Hotkey {self.config.hotkey} registered successfully")
                else:
                    print(f"Failed to register hotkey {self.config.hotkey}")
            
            # Register mouse button if enabled
            if self.config.enable_mouse_button:
                if MOUSE_AVAILABLE:
                    mouse_success = self.hotkey_manager.register_mouse_button("middle", self._toggle_recording)
                    if mouse_success:
                        print("Middle mouse button registered successfully")
                    else:
                        print("Failed to register middle mouse button")
                else:
                    print("Mouse button functionality requested but mouse library not available")
                    
        except Exception as e:
            print(f"Error setting up hotkeys: {e}")
    
    def _refresh_history(self):
        """Refresh the history display"""
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Add history items
        for entry in self.transcript_history.history:
            timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            preview = entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text']
            
            self.history_tree.insert('', 'end', values=(
                timestamp,
                entry['mode'],
                f"{entry['length']} chars",
                preview
            ))
    
    def _search_history(self, event=None):
        """Search transcript history"""
        query = self.search_var.get().strip()
        if not query:
            self._refresh_history()
            return
        
        # Clear existing items
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
        
        # Search and display results
        results = self.transcript_history.search_history(query)
        for entry in results:
            timestamp = entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            preview = entry['text'][:50] + "..." if len(entry['text']) > 50 else entry['text']
            
            self.history_tree.insert('', 'end', values=(
                timestamp,
                entry['mode'],
                f"{entry['length']} chars",
                preview
            ))
    
    def _clear_search(self):
        """Clear search and refresh history"""
        self.search_var.set("")
        self._refresh_history()
    
    def _view_transcript(self, event=None):
        """View full transcript on double-click"""
        self._view_selected_transcript()
    
    def _view_selected_transcript(self):
        """View selected transcript in detail"""
        selection = self.history_tree.selection()
        if not selection:
            return
        
        item = self.history_tree.item(selection[0])
        timestamp = item['values'][0]
        
        # Find the transcript entry
        for entry in self.transcript_history.history:
            if entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S') == timestamp:
                # Create new window to display full transcript
                detail_window = tk.Toplevel(self.root)
                detail_window.title(f"Transcript - {timestamp}")
                detail_window.geometry("800x600")
                
                # Apply theme
                detail_window.configure(bg=self.theme['bg'])
                
                # Create text widget
                text_widget = scrolledtext.ScrolledText(
                    detail_window,
                    wrap=tk.WORD,
                    font=('Consolas', 11),
                    bg=self.theme['text_bg'],
                    fg=self.theme['text_fg']
                )
                text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                
                # Insert transcript
                text_widget.insert(1.0, entry['text'])
                text_widget.config(state='disabled')  # Make read-only
                
                # Add buttons
                button_frame = ttk.Frame(detail_window)
                button_frame.pack(fill=tk.X, padx=10, pady=5)
                
                ttk.Button(
                    button_frame, 
                    text="Copy", 
                    command=lambda: pyperclip.copy(entry['text'])
                ).pack(side=tk.LEFT, padx=(0, 5))
                
                ttk.Button(
                    button_frame, 
                    text="Close", 
                    command=detail_window.destroy
                ).pack(side=tk.RIGHT)
                
                break
    
    def _copy_selected_transcript(self):
        """Copy selected transcript to clipboard"""
        selection = self.history_tree.selection()
        if not selection:
            return
        
        item = self.history_tree.item(selection[0])
        timestamp = item['values'][0]
        
        # Find and copy the transcript
        for entry in self.transcript_history.history:
            if entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S') == timestamp:
                try:
                    pyperclip.copy(entry['text'])
                    self.status_var.set("Transcript copied to clipboard")
                except Exception as e:
                    self.status_var.set(f"Clipboard error: {e}")
                break
    
    def _delete_selected_transcript(self):
        """Delete selected transcript from history"""
        selection = self.history_tree.selection()
        if not selection:
            return
        
        if messagebox.askyesno("Delete Transcript", "Are you sure you want to delete this transcript?"):
            item = self.history_tree.item(selection[0])
            timestamp = item['values'][0]
            
            # Find and remove the transcript
            for i, entry in enumerate(self.transcript_history.history):
                if entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S') == timestamp:
                    del self.transcript_history.history[i]
                    break
            
            self._refresh_history()
            self.status_var.set("Transcript deleted")
    
    def _export_all_history(self):
        """Export all transcript history"""
        if not self.transcript_history.history:
            messagebox.showwarning("Export", "No history to export")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Export History",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            format_type = "json" if filename.endswith('.json') else "txt"
            
            if self.transcript_history.export_history(filename, format_type):
                self.status_var.set(f"History exported to {filename}")
            else:
                messagebox.showerror("Export Error", "Failed to export history")
    
    def _clear_all_history(self):
        """Clear all transcript history"""
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all transcript history?"):
            self.transcript_history.history.clear()
            self._refresh_history()
            self.status_var.set("History cleared")
    
    def _import_audio(self):
        """Import and transcribe audio file"""
        filename = filedialog.askopenfilename(
            title="Import Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.m4a *.flac"),
                ("WAV files", "*.wav"),
                ("MP3 files", "*.mp3"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            # Process in background thread
            def process_import():
                try:
                    self.root.after(0, lambda: self.status_var.set("Importing audio..."))
                    
                    def progress_callback(message, progress):
                        self.root.after(0, lambda: self.status_var.set(message))
                        self.root.after(0, lambda: self.progress_var.set(progress))
                    
                    success, transcript = self.transcription_engine.transcribe_file(filename, progress_callback)
                    
                    if success:
                        self.root.after(0, lambda: self._import_complete(transcript, filename))
                    else:
                        self.root.after(0, lambda: messagebox.showerror("Import Error", f"Failed to transcribe audio: {transcript}"))
                        
                except Exception as e:
                    error_msg = f"Error importing audio: {e}"
                    self.root.after(0, lambda msg=error_msg: messagebox.showerror("Import Error", msg))
            
            threading.Thread(target=process_import, daemon=True).start()
    
    def _import_complete(self, transcript: str, filename: str):
        """Handle completed audio import"""
        # Update display
        self.transcription_display.delete(1.0, tk.END)
        self.transcription_display.insert(1.0, transcript)
        
        # Add to history
        mode_text = f"Imported from {os.path.basename(filename)}"
        self.transcript_history.add_transcript(transcript, mode=mode_text)
        
        # Update status
        self.status_var.set(f"Imported and transcribed {os.path.basename(filename)}")
        self.progress_var.set(0)
        
        # Refresh history display
        self._refresh_history()
    
    def _export_history(self):
        """Export transcript history via menu"""
        self._export_all_history()
    
    def _show_audio_devices(self):
        """Show available audio devices"""
        devices = self.audio_engine.get_audio_devices()
        
        if not devices:
            messagebox.showwarning("Audio Devices", "No audio input devices found")
            return
        
        # Create device selection window
        device_window = tk.Toplevel(self.root)
        device_window.title("Audio Devices")
        device_window.geometry("600x400")
        device_window.configure(bg=self.theme['bg'])
        
        # Create listbox for devices
        listbox_frame = ttk.Frame(device_window)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(listbox_frame, text="Available Audio Input Devices:").pack(anchor='w')
        
        device_listbox = tk.Listbox(
            listbox_frame,
            bg=self.theme['text_bg'],
            fg=self.theme['text_fg'],
            selectbackground=self.theme['select_bg'],
            selectforeground=self.theme['select_fg'],
            borderwidth=1,
            highlightthickness=0
        )
        device_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add devices to listbox
        for device in devices:
            device_listbox.insert(tk.END, f"{device['name']} (Index: {device['index']}, Channels: {device['channels']})")
        
        # Close button
        ttk.Button(device_window, text="Close", command=device_window.destroy).pack(pady=5)
    
    def _show_shortcuts(self):
        """Show keyboard shortcuts help"""
        shortcuts = [
            f"Recording: {self.config.hotkey.upper()} or Middle Mouse Button",
            "Copy Transcript: Ctrl+C (when transcript is selected)",
            "Clear Transcript: Ctrl+Delete",
            "Settings: Ctrl+,",
            "Export: Ctrl+E",
            "Exit: Ctrl+Q"
        ]
        
        messagebox.showinfo("Keyboard Shortcuts", "\n".join(shortcuts))
    
    def _show_about(self):
        """Show about dialog"""
        about_text = """Professional Speech-to-Text Application

Developed by Psynect Corp
www.psynect.ai

For support and issues, visit this GitHub repository

This advanced speech-to-text application provides high-accuracy transcription with intelligent context correction. Features multiple AI backends, persistent visual recording indicators, and seamless workflow integration.

🎯 Core Features:
• Multiple AI Backends: OpenAI (GPT-4o, Whisper-1), Groq Whisper 3 Large, Anthropic Claude
• Screenshot Context Correction: Uses vision models to improve transcription accuracy
• 7 Transcription Modes: High-accuracy, fast-processing, real-time, and transcription-only modes
• Persistent Recording Indicator: Always-on-top visual feedback during recording
• System Tray Integration: Background operation with tray controls
• Auto-paste Functionality: Seamless clipboard integration without focus stealing

🔧 Professional Features:
• Persistent Session History: Transcripts saved across app restarts with search
• Multiple Themes: Dark, Light, and Professional UI themes
• Global Controls: Middle mouse button (default) + Ctrl+Q hotkey for hands-free operation
• Configurable Settings: Audio devices, API keys, recording parameters
• Export/Import: Save transcripts and settings for backup/sharing
• Error Recovery: Robust clipboard handling and connection retry logic

🚀 Workflow Integration:
• Auto-minimize: Hide to tray during recording for distraction-free use
• Focus Preservation: Auto-paste without stealing focus from target applications
• Real-time Processing: Live transcription with instant visual feedback
• Cross-session Persistence: Settings and history maintained between sessions

Version 3.0 - GitHub Ready Edition"""
        
        messagebox.showinfo("About", about_text)
    
    def _open_settings(self):
        """Open settings tab - called from menu or tray"""
        # Switch to settings tab if notebook exists
        try:
            # Find the notebook widget and switch to settings tab
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Notebook):
                            child.select(1)  # Settings tab is usually index 1
                            break
        except Exception as e:
            print(f"Error opening settings: {e}")
    
    def _show_history(self):
        """Show history tab - called from menu or tray"""
        # Switch to history tab if notebook exists
        try:
            # Find the notebook widget and switch to history tab
            for widget in self.root.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Notebook):
                            child.select(2)  # History tab is usually index 2
                            break
        except Exception as e:
            print(f"Error showing history: {e}")
    
    def _on_window_close(self):
        """Handle window close event"""
        # Stop any active recording
        if self.is_recording:
            self._stop_recording()
        
        # Save current settings
        self.config.save_to_file()
        
        # Cleanup
        self._cleanup()
        
        # Destroy window
        self.root.destroy()
    
    def _cleanup(self):
        """Cleanup resources before exit"""
        # Stop system tray
        if self.system_tray:
            self.system_tray.stop_tray()
        
        # Unregister hotkeys
        self.hotkey_manager.unregister_all()
        
        # Cleanup audio engine
        if self.audio_engine:
            self.audio_engine.cleanup()
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            self._cleanup()


def main():
    """Main entry point"""
    try:
        # Initialize and run the GUI application
        app = SpeechToTextGUI()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
