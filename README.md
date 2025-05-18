# Speech-to-Text with Visual Context Correction
**by Psynect Corp**

This application allows you to transcribe your speech to text using OpenAI's models, while leveraging visual context from your screen to correct transcription errors. Simply press Ctrl+Q to start recording, and press Ctrl+Q again to stop recording and get your transcription directly copied to your clipboard and automatically pasted into your active text field.

## Features

- Five modes of operation: 
  - High-Accuracy Mode (slower, best quality using gpt-4o-transcribe + GPT-4.1)
  - Fast-Processing Mode (quicker using gpt-4o-mini models)
  - Real-time Mode (instant transcription without correction)
  - Transcription-Only (High-Accuracy) - Uses gpt-4o-transcribe without correction
  - Transcription-Only (Fast) - Uses gpt-4o-mini-transcribe without correction
- Start/stop recording with Ctrl+Q
- Visual recording indicator (mouse pointer animation during recording)
- Live transcription display window (in Real-time Mode)
- Automatic screenshot capture from the monitor with your mouse cursor (in modes with correction)
- Transcription error correction using visual context (with Claude Sonnet as backup)
- Handles large audio files by chunking them into smaller pieces
- Transcribed text is automatically copied to clipboard AND pasted
- Support for multi-monitor setups

## Models Used

This application uses the following OpenAI models:

| Mode | Transcription Model | Correction Model |
|------|---------------------|------------------|
| 1. High-Accuracy | gpt-4o-transcribe | GPT-4.1 |
| 2. Fast-Processing | gpt-4o-mini-transcribe | gpt-4o-mini |
| 3. Real-time | gpt-4o-transcribe | None (no correction) |
| 4. Transcription-Only (High-Accuracy) | gpt-4o-transcribe | None (no correction) |
| 5. Transcription-Only (Fast) | gpt-4o-mini-transcribe | None (no correction) |

Fallback models:
- Transcription fallback: Groq with Whisper-large-v3 (if configured)
- Correction fallback: Claude 3.5 Sonnet (if configured)

## Requirements

- Python 3.8 or higher
- An OpenAI API key (required)
- Groq API key (optional, used as fallback for transcription)
- Anthropic API key (optional, used as fallback for correction)

## Quick Start (Windows)

For the easiest setup on Windows, simply run:

1. `step1_install.bat` - Sets up the environment and dependencies
2. `step2_run.bat` - Runs the application

## Installation

1. Clone the repository to your machine
2. Run `step1_install.bat` to set up the environment
3. Run `step2_run.bat` to start the application

## Usage

1. When prompted, select one of the five modes (or press Enter to select High-Accuracy Mode):

   **With Screenshot Correction:**
   - Option 1: High-Accuracy Mode (gpt-4o-transcribe + GPT-4.1) - Best quality, but slower
   - Option 2: Fast-Processing Mode (gpt-4o-mini models) - Quicker results, best for clear speakers
   
   **Without Correction (Transcription Only):**
   - Option 3: Real-time Mode (gpt-4o-transcribe, no correction) - Instant transcription
   - Option 4: Transcription-Only (High-Accuracy) - Uses gpt-4o-transcribe, no correction
   - Option 5: Transcription-Only (Fast) - Uses gpt-4o-mini-transcribe, no correction

2. Press Ctrl+Q to start recording your speech:
   - In screenshot modes (1 & 2), this will capture a screenshot of your current screen
   - In Real-time Mode (3), a small floating window will appear showing the live transcription
   - In Transcription-Only modes (4 & 5), no screenshot is captured

3. Press Ctrl+Q again to stop recording and get the transcription:
   - In modes with screenshot correction (1 & 2), the audio will be transcribed and corrected
   - In modes without correction (3, 4, 5), the transcription is returned without further processing

4. The transcription will automatically be copied to your clipboard and pasted wherever your cursor is.

5. To exit the application, press Ctrl+C in the terminal window.

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

## Multi-Monitor Support

The application automatically detects which monitor your mouse cursor is on when you start recording. It will capture a screenshot of that specific monitor (in modes with correction), ensuring you get the visual context that's most relevant to what you're looking at or working on.

## Screenshot Management

The application uses a simple approach to manage screenshots:

1. Each time you start recording in screenshot correction modes (1 & 2), a new screenshot is captured
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

When API keys are provided, in correction modes (1 & 2) the application will:
1. Send the initial transcription to the correction model (GPT-4.1 or gpt-4o-mini, depending on mode)
2. Include the screenshot as visual context
3. Ask the model to correct any errors in the transcription based on the visual information
4. Use a carefully crafted system prompt to ensure the model:
   - Corrects technical terms, acronyms, or specialized vocabulary visible in the screenshot
   - Improves grammar and readability
   - Preserves the original meaning of the transcription

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

- If screenshot capture fails:
  - Check that you have the required permissions for screen capture
  - Verify that the `mss` and `pyautogui` packages are installed correctly

- If correction isn't working:
  - Verify your OpenAI API key is correct in the `.env` file for GPT-4.1
  - If using Claude as backup, verify your Anthropic API key is correct
  - Ensure you have sufficient API credits for vision requests

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