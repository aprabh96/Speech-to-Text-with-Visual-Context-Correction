# Speech-to-Text with Visual Context Correction
**by Psynect Corp**

This application allows you to transcribe your speech to text using OpenAI's models or Groq's Whisper 3 Large, while leveraging visual context from your screen to correct transcription errors. Simply press Ctrl+Q or click your middle mouse button (scroll wheel) to start recording, speak naturally, then click again to stop and get your transcription automatically copied to your clipboard and pasted into your active text field.

**Perfect for keyboard-free operation** - work entirely with your mouse for seamless voice-to-text workflows without ever touching the keyboard.

## Features

- Multiple transcription backends:
  - **OpenAI GPT-4o family** - Fastest & most accurate, with 5 operation modes
  - **OpenAI Whisper-1** - Rock-solid stability for long recordings
  - **Groq Whisper 3 Large** - Fast, accurate alternative with optional correction
- Operation modes available:
  - High-Accuracy Mode (best quality using gpt-4o-transcribe + GPT-4.1)
  - Fast-Processing Mode (quicker using gpt-4o-mini models)
  - Real-time Mode (instant transcription without correction)
  - Transcription-Only modes (with or without correction)
  - Groq Whisper 3 Large (with or without screenshot correction)
- **Safety Features:**
  - **Recording Duration Limit** - Configurable safety limit (default: 5 minutes) to prevent accidental long recordings that waste API credits
  - **Settings Menu** - Easy access to configure safety limits and other options
- Start/stop recording with Ctrl+Q or middle mouse button (scroll wheel click)
- Visual recording indicator (mouse pointer animation during recording)
- Live transcription display window (in Real-time Mode)
- Automatic screenshot capture from the monitor with your mouse cursor (in modes with correction)
- Transcription error correction using visual context (with Claude Sonnet as backup)
- Handles large audio files by chunking them into smaller pieces
- Transcribed text is automatically copied to clipboard AND pasted
- Support for multi-monitor setups

## Keyboard-Free Operation

One of the key benefits of the **middle mouse button support** is enabling completely keyboard-free operation during transcription workflows. 

### The Problem
Many users found themselves constantly reaching for the keyboard just to press Ctrl+Q to start and stop recordings, while spending the rest of their time using only the mouse for navigation, clicking, and positioning the cursor. This created an unnecessary dependency on the keyboard for what should be a seamless voice-to-text workflow.

### The Solution
With middle mouse button (scroll wheel click) support, you can now:

- **Work entirely with just your mouse** - no need to touch the keyboard at all
- **Keep your hands in the optimal position** - maintain your natural mouse grip without reaching for keys
- **Streamline your workflow** - click the scroll wheel to start recording, speak your text, click again to stop and automatically paste
- **Stay focused on your content** - eliminate the context switching between mouse and keyboard

### Typical Keyboard-Free Workflow
1. Position your cursor where you want text to appear (mouse only)
2. Click the middle mouse button to start recording
3. Speak your content naturally
4. Click the middle mouse button again to stop recording
5. Text is automatically transcribed and pasted at your cursor location
6. Continue working with your mouse - no keyboard interaction needed

This is especially valuable for:
- **Content creators** writing long documents or emails
- **Professionals** who dictate notes while reviewing documents on screen
- **Users with accessibility needs** who prefer mouse-only operation
- **Anyone seeking a more efficient voice-to-text workflow**

The traditional Ctrl+Q hotkey remains available as an alternative for users who prefer keyboard shortcuts.

## Models Used

This application supports multiple transcription backends:

### Backend Options:
1. **GPT-4o family** - OpenAI's latest models (fastest & most accurate)
2. **Whisper-1** - OpenAI's stable model (rock-solid reliability)
3. **Groq Whisper 3 Large with correction** - Fast alternative with screenshot context
4. **Groq Whisper 3 Large without correction** - Fast transcription-only mode

### Detailed Model Usage:

| Backend | Mode | Transcription Model | Correction Model |
|---------|------|---------------------|------------------|
| GPT-4o | 1. High-Accuracy | gpt-4o-transcribe | GPT-4.1 |
| GPT-4o | 2. Fast-Processing | gpt-4o-mini-transcribe | gpt-4o-mini |
| GPT-4o | 3. Real-time | gpt-4o-transcribe | None |
| GPT-4o | 4. Transcription-Only (High) | gpt-4o-transcribe | None |
| GPT-4o | 5. Transcription-Only (Fast) | gpt-4o-mini-transcribe | None |
| Whisper-1 | With correction | whisper-1 | GPT-4.1 |
| Whisper-1 | Without correction | whisper-1 | None |
| Groq | With correction | whisper-large-v3 | GPT-4.1 |
| Groq | Without correction | whisper-large-v3 | None |

### Fallback Models:
- Transcription fallback: Groq Whisper-large-v3 ↔ OpenAI models (cross-fallback)
- Correction fallback: Claude 3.5 Sonnet (if configured)

## Requirements

- Python 3.8 or higher
- **API Keys (at least one required):**
  - OpenAI API key (required for GPT-4o and Whisper-1 backends, vision correction)
  - Groq API key (required for Groq Whisper 3 Large backend)
  - Anthropic API key (optional, used as fallback for correction)

## Quick Start (Windows)

For the easiest setup on Windows, simply run:

1. `step1_install.bat` - Sets up the environment and dependencies
2. `step2_run.bat` - Runs the application

## Installation

1. Clone the repository to your machine
2. Run `step1_install.bat` to set up the environment
3. **Create a `.env` file** in the project directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```
   **Important:** 
   - You need at least one of OPENAI_API_KEY or GROQ_API_KEY to run the application
   - ANTHROPIC_API_KEY is optional for Claude fallback correction
   - Get your OpenAI API key from: https://platform.openai.com/api-keys
   - Get your Groq API key from: https://console.groq.com/keys
   - Get your Anthropic API key from: https://console.anthropic.com/
4. Run `step2_run.bat` to start the application

## Usage

1. **First, select your transcription backend** (or press Enter for default GPT-4o):
   - **Option 1: GPT-4o family** - Fastest & most accurate, but may truncate very long recordings
   - **Option 2: Whisper-1** - Rock-solid stability, recommended for long recordings
   - **Option 3: Groq Whisper 3 Large with correction** - Fast, accurate, with screenshot context
   - **Option 4: Groq Whisper 3 Large without correction** - Fast, accurate, transcription only
   - **Option S: Settings** - Configure safety limits and other options

2. **Then select operation mode** (varies by backend selected):

   **For GPT-4o backend (Option 1):**
   - Option 1: High-Accuracy Mode (gpt-4o-transcribe + GPT-4.1) - Best quality, slower
   - Option 2: Fast-Processing Mode (gpt-4o-mini models) - Quicker results
   - Option 3: Real-time Mode (instant transcription, no correction)
   - Option 4: Transcription-Only (High-Accuracy)
   - Option 5: Transcription-Only (Fast)

   **For Whisper-1 backend (Option 2):**
   - Option 1: With Screenshot Correction (whisper-1 + GPT-4.1)
   - Option 2: Transcription Only (whisper-1 without correction)

   **For Groq backends (Options 3 & 4):**
   - Mode is automatically set based on your backend choice

3. Press Ctrl+Q or click the middle mouse button (scroll wheel) to start recording your speech:
   - In modes with screenshot correction, this will capture a screenshot of your current screen
   - In Real-time Mode, a small floating window will appear showing the live transcription
   - In transcription-only modes, no screenshot is captured

4. Press Ctrl+Q or click the middle mouse button again to stop recording and get the transcription:
   - In modes with screenshot correction, the audio will be transcribed and corrected using visual context
   - In modes without correction, the transcription is returned without further processing

5. The transcription will automatically be copied to your clipboard and pasted wherever your cursor is.

6. To exit the application, press Ctrl+C in the terminal window.

## Safety Features

### Recording Duration Limit
The application includes a built-in safety feature to prevent accidental long recordings that could consume excessive API credits:

- **Default limit:** 5 minutes per recording
- **Configurable:** Easily adjustable from 10 seconds to 60 minutes
- **Smart warnings:** Shows duration info and gives options when limit is exceeded
- **Cost protection:** Option to cancel transcription for long recordings

### How it works:
1. **During recording:** Shows the current safety limit when you start recording
2. **When stopping:** Displays actual recording duration
3. **If exceeded:** Offers three options:
   - Transcribe anyway (uses API credits)
   - Cancel transcription (saves costs)
   - Change safety settings

### Settings Menu
Access the settings menu by selecting "S" when choosing your backend:

- **Change Maximum Recording Duration:** Set custom time limits
- **Reset to Default:** Return to 5-minute default
- **Real-time configuration:** No need to restart the application

**Example scenarios this prevents:**
- Accidentally leaving recording on overnight
- Forgetting to stop recording during long meetings
- Unintended recordings from accidental hotkey presses

## How It Works

### High-Accuracy Mode (Option 1)
1. Records audio from your default microphone
2. Captures a screenshot of the monitor where your mouse cursor is located
3. Changes the mouse pointer to an animated cursor to indicate recording is active
4. When recording stops:
   - Saves the audio to a file
   - Checks if the file is within size limits
   - If the file is too large, it chunks the audio into smaller pieces with overlapping segments
   - Sends the audio file(s) to OpenAI's gpt-4o-transcribe API for transcription (with Groq as fallback)
   - Combines the transcriptions if necessary
   - Sends the transcription and screenshot to GPT-4.1 (or Claude as backup) for correction
   - Copies the final transcription to your clipboard and pastes it

### Fast-Processing Mode (Option 2)
Works exactly like the High-Accuracy mode but uses:
- gpt-4o-mini-transcribe for faster transcription
- gpt-4o-mini for faster correction with the screenshot
This option is best for users who speak clearly without strong accents and prioritize speed over absolute accuracy.

### Real-time Mode (Option 3)
1. Records audio from your default microphone
2. Streams the audio in real-time to OpenAI's WebSocket API
3. Displays the transcription as it happens in a small floating window above your cursor
4. Changes the mouse pointer to an animated cursor to indicate recording is active
5. When recording stops:
   - Finalizes the transcription
   - Copies the final transcription to your clipboard and pastes it

### Transcription-Only (High-Accuracy) Mode (Option 4)
1. Records audio from your default microphone 
2. Changes the mouse pointer to an animated cursor to indicate recording is active
3. When recording stops:
   - Saves the audio to a file
   - Transcribes using OpenAI's gpt-4o-transcribe model for high-quality results
   - Copies the transcription directly to your clipboard and pastes it
   - No screenshot is captured and no correction is performed

### Transcription-Only (Fast) Mode (Option 5)
Same as Option 4, but uses the faster gpt-4o-mini-transcribe model for quicker processing when accuracy is less critical.

### Groq Whisper 3 Large with Screenshot Correction (Backend Option 3)
1. Records audio from your default microphone
2. Captures a screenshot of the monitor where your mouse cursor is located
3. Changes the mouse pointer to an animated cursor to indicate recording is active
4. When recording stops:
   - Saves the audio to a file
   - Sends the audio file to Groq's Whisper-large-v3 API for fast, accurate transcription
   - Sends the transcription and screenshot to GPT-4.1 for correction using visual context
   - Falls back to OpenAI transcription if Groq fails
   - Copies the final corrected transcription to your clipboard and pastes it

### Groq Whisper 3 Large Transcription-Only (Backend Option 4)
1. Records audio from your default microphone
2. Changes the mouse pointer to an animated cursor to indicate recording is active
3. When recording stops:
   - Saves the audio to a file
   - Sends the audio file to Groq's Whisper-large-v3 API for fast, accurate transcription
   - Falls back to OpenAI transcription if Groq fails
   - Copies the transcription directly to your clipboard and pastes it
   - No screenshot is captured and no correction is performed

## Multi-Monitor Support

The application automatically detects which monitor your mouse cursor is on when you start recording. It will capture a screenshot of that specific monitor (in modes with correction), ensuring you get the visual context that's most relevant to what you're looking at or working on.

## Screenshot Management

The application uses a simple approach to manage screenshots:

1. Each time you start recording in screenshot correction modes, a new screenshot is captured
   - GPT-4o High-Accuracy and Fast-Processing modes
   - Whisper-1 with correction mode
   - Groq Whisper 3 Large with correction mode
2. The screenshot is always saved to the same file (`latest_screenshot.jpg`) in the `screenshots` folder
3. Each new screenshot automatically overwrites the previous one
4. This ensures only one screenshot file is stored at any time

This minimalist approach saves disk space while still allowing you to inspect the most recent screenshot if needed for troubleshooting.

## Screenshot Inspection

The most recent screenshot is saved as `latest_screenshot.jpg` in the `screenshots` folder. This allows you to:

1. Check what visual context was sent to the correction model for the last transcription
2. Troubleshoot issues with transcription correction
3. Verify the correct monitor was captured

The application will print the full path to the saved screenshot in the console when debug mode is enabled.

## Transcription Correction

When API keys are provided, in correction modes the application will:
1. Send the initial transcription to the correction model (GPT-4.1, gpt-4o-mini, or Claude, depending on mode and availability)
2. Include the screenshot as visual context
3. Ask the model to correct any errors in the transcription based on the visual information
4. Use a carefully crafted system prompt to ensure the model:
   - Corrects technical terms, acronyms, or specialized vocabulary visible in the screenshot
   - Improves grammar and readability
   - Preserves the original meaning of the transcription

**Correction modes include:**
- GPT-4o High-Accuracy and Fast-Processing modes
- Whisper-1 with correction mode
- Groq Whisper 3 Large with correction mode

This feature is particularly useful for:
- Technical discussions where specialized terminology might be misheard
- Situations where visual information can disambiguate similar-sounding words
- Capturing accurate references to on-screen text, data, or UI elements

## Visual Recording Indicator

To help you easily identify when recording is active, the application shows:

1. An animated "thinking" cursor while recording (in all modes)
2. A live transcription window (in Real-time Mode)

If for any reason the application closes unexpectedly, it will automatically restore your normal mouse pointer.

## Troubleshooting

- If you encounter issues with PyAudio installation, you may need to install PortAudio first.
  - On Windows: `pip install pipwin` followed by `pipwin install pyaudio`
  - On macOS: `brew install portaudio` followed by `pip install pyaudio`
  - On Linux: `sudo apt-get install python3-pyaudio` or equivalent for your distribution

- If the application doesn't respond to Ctrl+Q, try these fixes:
  - Make sure no other application is capturing this key combination
  - Run the application as administrator
  - Try using the `setup_and_run.bat` script which includes more robust error handling

- If the middle mouse button doesn't work:
  - Ensure your mouse has a working scroll wheel that can be clicked
  - Some applications may capture middle mouse button clicks - try closing other applications
  - Use Ctrl+Q as an alternative if middle mouse button is not responding
  - Run the application as administrator if mouse events are being blocked
  - **Note**: The middle mouse button enables completely keyboard-free operation - ideal for workflows where you only want to use your mouse

- If screenshot capture fails:
  - Check that you have the required permissions for screen capture
  - Verify that the `mss` and `pyautogui` packages are installed correctly

- If correction isn't working:
  - Verify your OpenAI API key is correct in the `.env` file for GPT-4.1
  - If using Claude as backup, verify your Anthropic API key is correct
  - Ensure you have sufficient API credits for vision requests

- If Groq transcription fails:
  - Verify your Groq API key is correct in the `.env` file (GROQ_API_KEY)
  - Check your Groq API credits and rate limits
  - The application will automatically fall back to OpenAI transcription if Groq fails

- If auto-pasting doesn't work:
  - Make sure you have clicked into a text field before stopping the recording
  - Check that no application is blocking the Ctrl+V shortcut
  - You can still manually paste using Ctrl+V as the text is already in your clipboard

## License

This software is open source and free to use for non-commercial purposes. Commercial use is prohibited without express permission from Psynect Corp.

## Credits

Developed by [Psynect Corp](https://psynect.ai)

---

© Psynect Corp. All rights reserved. 