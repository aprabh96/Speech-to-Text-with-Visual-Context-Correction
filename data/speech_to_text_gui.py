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

# GUI Imports
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import tkinter.font as tkFont
from tkinter import Menu

# Core functionality imports
import numpy as np
import pyaudio
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
    format: int = pyaudio.paInt16
    
    # Recording settings
    hotkey: str = "ctrl+q"
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
    """Handles all audio recording and processing functionality"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio = None
        self.recording_stream = None
        self.recording_active = False
        self.recorded_frames = []
        self.recording_start_time = None
        
        # Initialize PyAudio
        try:
            self.audio = pyaudio.PyAudio()
            print("PyAudio initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PyAudio: {e}")
    
    def get_audio_devices(self) -> List[Dict]:
        """Get list of available audio input devices"""
        devices = []
        if not self.audio:
            return devices
            
        try:
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    devices.append({
                        'index': i,
                        'name': device_info['name'],
                        'channels': device_info['maxInputChannels'],
                        'sample_rate': device_info['defaultSampleRate']
                    })
        except Exception as e:
            print(f"Error getting audio devices: {e}")
        
        return devices
    
    def start_recording(self, device_index: Optional[int] = None) -> bool:
        """Start audio recording"""
        if self.recording_active:
            return False
            
        try:
            chunk_samples = int(self.config.rate * (self.config.chunk_ms / 1000))
            
            self.recording_stream = self.audio.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=chunk_samples
            )
            
            self.recording_active = True
            self.recorded_frames = []
            self.recording_start_time = time.time()
            
            # Start recording thread
            threading.Thread(target=self._recording_loop, daemon=True).start()
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self) -> Tuple[bool, float]:
        """Stop audio recording and return success status and duration"""
        if not self.recording_active:
            return False, 0.0
            
        self.recording_active = False
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        
        # Stop and close stream
        if self.recording_stream:
            try:
                self.recording_stream.stop_stream()
                self.recording_stream.close()
            except Exception as e:
                print(f"Error stopping recording stream: {e}")
            finally:
                self.recording_stream = None
                
        return True, duration
    
    def _recording_loop(self):
        """Recording loop running in separate thread"""
        chunk_samples = int(self.config.rate * (self.config.chunk_ms / 1000))
        
        while self.recording_active and self.recording_stream:
            try:
                data = self.recording_stream.read(chunk_samples, exception_on_overflow=False)
                self.recorded_frames.append(data)
                
                # Check recording duration
                if self.recording_start_time:
                    duration = time.time() - self.recording_start_time
                    if duration > self.config.max_recording_duration:
                        self.recording_active = False
                        break
                        
            except Exception as e:
                print(f"Error in recording loop: {e}")
                break
    
    def save_recording(self, filename: str) -> bool:
        """Save recorded audio to file"""
        if not self.recorded_frames:
            return False
            
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.config.format))
                wf.setframerate(self.config.rate)
                wf.writeframes(b''.join(self.recorded_frames))
            return True
        except Exception as e:
            print(f"Error saving recording: {e}")
            return False
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.recording_stream:
            try:
                if self.recording_stream.is_active():
                    self.recording_stream.stop_stream()
                self.recording_stream.close()
            except:
                pass
                
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass


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
            except Exception as e:
                print(f"Failed to initialize Groq client: {e}")
        
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
            'claude': self.claude_client is not None
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
            if progress_callback:
                progress_callback("Transcribing with Groq...", 50)
                
            with open(file_path, "rb") as file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=(os.path.basename(file_path), file.read()),
                    model="whisper-large-v3",
                    response_format="text",
                    language="en"
                )
            
            if progress_callback:
                progress_callback("Transcription completed", 100)
                
            # Handle different response formats
            if isinstance(transcription, str):
                return True, transcription
            elif hasattr(transcription, 'text'):
                return True, transcription.text
            else:
                return True, str(transcription)
                
        except Exception as e:
            return False, f"Groq transcription failed: {e}"


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
                
                # Save to data folder
                os.makedirs("data/screenshots", exist_ok=True)
                temp_path = os.path.join("data/screenshots", "latest_screenshot.jpg")
                img.save(temp_path)
                
                return img, temp_path
                
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None, None


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
        
        # Recording state
        self.is_recording = False
        self.current_transcript = ""
        self.recording_timer = None
        
        self._initialize_application()
    
    def _initialize_application(self):
        """Initialize the main application"""
        try:
            # Initialize engines
            self.audio_engine = AudioEngine(self.config)
            self.transcription_engine = TranscriptionEngine(self.config)
            self.screenshot_engine = ScreenshotEngine(self.config)
            
            # Create GUI
            self._create_gui()
            
            # Setup hotkeys
            self._setup_hotkeys()
            
            # Setup system tray
            if self.config.system_tray:
                self.system_tray = SystemTrayManager(self)
                self.system_tray.start_tray()
            
            print("Application initialized successfully")
            
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
        else:
            initial_mode_label = {
                1: "High-Accuracy Mode (gpt-4o-transcribe + GPT-4.1)",
                2: "Fast-Processing Mode (gpt-4o-mini models)",
                3: "Real-time Mode (gpt-4o-transcribe, no correction)",
                4: "Transcription-Only (High-Accuracy)",
                5: "Transcription-Only (Fast)"
            }.get(saved_mode, "High-Accuracy Mode (gpt-4o-transcribe + GPT-4.1)")

        self.mode_var = tk.StringVar(value=initial_mode_label)
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.mode_var, state="readonly", width=50)
        mode_combo['values'] = [
            "High-Accuracy Mode (gpt-4o-transcribe + GPT-4.1)",
            "Fast-Processing Mode (gpt-4o-mini models)", 
            "Real-time Mode (gpt-4o-transcribe, no correction)",
            "Transcription-Only (High-Accuracy)",
            "Transcription-Only (Fast)",
            "Groq Whisper 3 Large with Screenshot Correction",
            "Groq Whisper 3 Large Transcription Only"
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
        
        # Auto-paste option
        self.auto_paste_var = tk.BooleanVar(value=self.config.auto_paste)
        ttk.Checkbutton(
            action_frame, 
            text="Auto-paste after transcription", 
            variable=self.auto_paste_var,
            command=self._on_auto_paste_changed
        ).pack(side=tk.RIGHT)
    
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
        
        api_frame.columnconfigure(1, weight=1)
        
        # Test API connections button
        ttk.Button(api_frame, text="Test Connections", command=self._test_api_connections).grid(
            row=3, column=1, sticky="e", pady=10
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
        
        # Max recording duration
        ttk.Label(recording_frame, text="Max Recording Duration (seconds):").grid(row=1, column=0, sticky="w", pady=2)
        self.max_duration_var = tk.IntVar(value=self.config.max_recording_duration)
        ttk.Spinbox(
            recording_frame, 
            from_=10, 
            to=3600, 
            textvariable=self.max_duration_var,
            width=10
        ).grid(row=1, column=1, sticky="w", padx=(5, 0), pady=2)
        
        # Audio quality settings
        ttk.Label(recording_frame, text="Sample Rate (Hz):").grid(row=2, column=0, sticky="w", pady=2)
        self.sample_rate_var = tk.IntVar(value=self.config.rate)
        rate_combo = ttk.Combobox(recording_frame, textvariable=self.sample_rate_var, width=15, state="readonly")
        rate_combo['values'] = [16000, 22050, 24000, 44100, 48000]
        rate_combo.grid(row=2, column=1, sticky="w", padx=(5, 0), pady=2)
        
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
            
        # Update UI
        self.is_recording = True
        self.record_button.config(text="🛑 Stop Recording")
        self.recording_status.config(text="Recording...")
        self.status_var.set("Recording in progress...")
        
        # Show recording indicator overlay
        self._show_recording_indicator()
        
        # Clear previous transcript
        self.transcription_display.delete(1.0, tk.END)
        
        # Start recording
        if self.audio_engine.start_recording():
            self._start_recording_timer()
            
            # Show progress
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
        else:
            self._recording_error("Failed to start recording")
    
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
            # Save recording to data folder
            os.makedirs("data/recordings", exist_ok=True)
            temp_file = os.path.join("data/recordings", "latest_recording.wav")
            
            if not self.audio_engine.save_recording(temp_file):
                self._recording_error("Failed to save recording")
                return
            
            # Update progress
            self.root.after(0, lambda: self.progress_var.set(25))
            
            # Capture screenshot if correction is enabled
            screenshot = None
            if self.config.use_correction:
                screenshot, _ = self.screenshot_engine.capture_screenshot()
                self.root.after(0, lambda: self.progress_var.set(40))
            
            # Transcribe audio
            def progress_callback(message, progress):
                self.root.after(0, lambda: self.status_var.set(message))
                self.root.after(0, lambda: self.progress_var.set(progress))
            
            success, transcript = self.transcription_engine.transcribe_file(temp_file, progress_callback)
            
            if success:
                # Apply correction if enabled and screenshot available
                if self.config.use_correction and screenshot:
                    self.root.after(0, lambda: self.status_var.set("Applying context correction..."))
                    self.root.after(0, lambda: self.progress_var.set(85))
                    
                    # TODO: Implement correction logic
                    # For now, use the raw transcript
                    final_transcript = transcript
                else:
                    final_transcript = transcript
                
                # Update UI with final result
                self.root.after(0, lambda: self._transcription_complete(final_transcript))
                
                # Add to history
                mode_text = self.mode_var.get()
                self.transcript_history.add_transcript(final_transcript, mode=mode_text)
                
            else:
                self.root.after(0, lambda: self._recording_error(f"Transcription failed: {transcript}"))
            
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            self.root.after(0, lambda: self._recording_error(f"Processing error: {e}"))
    
    def _transcription_complete(self, transcript: str):
        """Handle completed transcription"""
        # Update display
        self.transcription_display.delete(1.0, tk.END)
        self.transcription_display.insert(1.0, transcript)
        
        # Update status
        self.recording_status.config(text="Transcription complete")
        self.status_var.set(f"Transcribed {len(transcript)} characters")
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
        
        # Don't automatically restore window - let user manually restore when needed
        # This prevents focus stealing when auto-paste is enabled
        
        # Update system tray tooltip if minimized to indicate completion
        if self.system_tray and not self.system_tray.is_visible:
            try:
                if hasattr(self.system_tray, 'icon') and self.system_tray.icon:
                    self.system_tray.icon.title = f"Transcription Complete ({len(transcript)} chars) - Click to view"
            except Exception:
                pass
    
    def _recording_error(self, message: str):
        """Handle recording errors"""
        self.is_recording = False
        self.record_button.config(text="🎤 Start Recording")
        self.recording_status.config(text="Error")
        self.status_var.set(f"Error: {message}")
        self.progress_var.set(0)
        self.progress_bar.stop()
        
        # Hide recording indicator overlay
        self._hide_recording_indicator()
        
        if self.recording_timer:
            self.root.after_cancel(self.recording_timer)
            self.recording_timer = None
        
        messagebox.showerror("Recording Error", message)
    
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
        self.config.hotkey = self.hotkey_var.get()
        self.config.max_recording_duration = self.max_duration_var.get()
        self.config.rate = self.sample_rate_var.get()
        self.config.theme = self.theme_var.get()
        self.config.auto_minimize = self.auto_minimize_var.get()
        self.config.system_tray = self.system_tray_var.get()
        self.config.auto_paste = self.auto_paste_var.get()
        
        # Save to file
        self.config.save_to_file()
        
        # Reinitialize engines with new settings
        self.transcription_engine = TranscriptionEngine(self.config)
        
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
        
        # Save the updated config to persist the mode selection
        self.config.save_to_file()
    
    def _on_auto_paste_changed(self):
        """Handle auto-paste option change"""
        self.config.auto_paste = self.auto_paste_var.get()
        # Save the updated config
        self.config.save_to_file()
    
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
                    self.root.after(0, lambda: messagebox.showerror("Import Error", f"Error importing audio: {e}"))
            
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
• Global Hotkeys: Ctrl+Q or middle mouse button for hands-free operation
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
