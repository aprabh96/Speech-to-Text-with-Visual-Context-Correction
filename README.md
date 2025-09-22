# Professional Speech-to-Text Application

**Developed by Psynect Corp** - [www.psynect.ai](https://www.psynect.ai)

An advanced speech-to-text application with both **CLI** and **GUI** interfaces that provides high-accuracy transcription with intelligent context correction. Features multiple AI backends, middle mouse button recording (default), persistent visual indicators, and seamless workflow integration for professional use.

## 🚀 Quick Start

### GUI Application (Recommended)
```bash
# Run the all-in-one launcher
run.bat
```

### Command Line Interface  
```bash
# Run the original CLI version
python speech_to_text.py
```

## 📋 Features

### Core Functionality
- **Multiple AI Backends**: OpenAI (GPT-4o, Whisper-1), Groq Whisper 3 Large
- **Screenshot Context Correction**: Uses vision models (GPT-4.1, Claude) to improve accuracy
- **Global Controls**: Middle mouse button (default) + Ctrl+Q hotkey for hands-free operation
- **Auto-paste**: Automatic clipboard copy and paste functionality

### GUI Application Features
- **Professional Interface**: Modern desktop application with themes
- **Persistent History**: Transcript history across sessions with search
- **System Tray Integration**: Background operation with tray menu
- **Multiple Themes**: Dark, Light, and Professional themes
- **Settings Management**: Comprehensive configuration interface

### CLI Application Features
- **Live Transcription Display**: Real-time transcription window
- **Multiple Correction Models**: GPT-4.1 and Claude fallback
- **Safety Features**: Recording duration limits
- **Buffered and Live Modes**: Choose your workflow

## 📁 Project Structure

```
SpeechToTextWithScreenContext/
├── run.bat                    # Main launcher (GUI)
├── speech_to_text.py         # CLI application
├── README.md                 # This file
├── requirements.txt          # CLI dependencies
└── data/                     # Application data
    ├── .env                  # API credentials (create this)
    ├── speech_to_text_gui.py # GUI application
    ├── requirements_gui.txt  # GUI dependencies
    ├── app_config.json       # GUI settings (auto-created)
    ├── transcript_history.json # History (auto-created)
    ├── screenshots/          # Screenshot storage
    └── recordings/           # Audio recordings
```

## ⚙️ Installation & Setup

### Quick Setup
1. **Run the launcher** - It handles everything automatically:
   ```bash
   run.bat
   ```

2. **Configure API keys** (first run):
   - Edit `data\.env` file, or
   - Use the GUI settings panel

### API Keys Required
- **OpenAI**: For GPT-4o transcription and GPT-4.1 correction
  - Get your key: https://platform.openai.com/api-keys
- **Groq** (Optional): For Whisper 3 Large transcription  
  - Get your key: https://console.groq.com/keys
- **Anthropic** (Optional): For Claude correction fallback
  - Get your key: https://console.anthropic.com/

## 🎯 Usage

### Recording Controls
- **Middle Mouse Button**: Click scroll wheel to record (enabled by default)
- **Hotkey**: Ctrl+Q keyboard shortcut (always available)
- **GUI Button**: Record/Stop button in interface
- **Settings**: Both recording methods can be configured in Settings tab

### Transcription Modes
1. **High-Accuracy Mode** (gpt-4o-transcribe + GPT-4.1)
2. **Fast-Processing Mode** (gpt-4o-mini models)
3. **Real-time Mode** (gpt-4o-transcribe, no correction)
4. **Transcription-Only** modes
5. **Groq Whisper 3 Large** options

## 🎨 Key Features

### Perfect for Keyboard-Free Operation
- Work entirely with mouse for seamless voice-to-text workflows
- Global hotkeys and mouse button support
- Automatic clipboard and paste operations

### Professional Grade
- Multiple AI model support with fallbacks
- Screenshot-based context correction
- Safety features and error handling
- Persistent settings and history

## 📞 Support

For support, issues, or feature requests:
- Create an issue on this GitHub repository
- Check existing issues and discussions

## 📄 License

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

✅ **You can**: Use, share, and modify the software for personal and educational purposes  
✅ **You must**: Give credit and share any modifications under the same license  
❌ **You cannot**: Use this software for commercial purposes  

See the [LICENSE](LICENSE) file for full details.

Professional Speech-to-Text Application by Psynect Corp.
