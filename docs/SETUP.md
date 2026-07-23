# Setup

## macOS (verified)
1. Double-click **First-time setup.command**. If macOS warns "unidentified developer",
   right-click → Open. It installs ffmpeg + BlackHole, builds the Python 3.11 env,
   and downloads/converts the models (a few GB, one time).
2. Speaker labels need a free HuggingFace token:
   - Create one at https://huggingface.co/settings/tokens
   - Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1
   - Put it in `config.toml` as `hf_token = "hf_..."` (copy from `config.example.toml`),
     or export `HF_TOKEN`.
3. Double-click **Start Transcrb.command** — your browser opens the UI.

## System audio (both sides of a call)
After BlackHole is installed, create a Multi-Output Device (Audio MIDI Setup) that
includes BlackHole, route your call audio to it, and pick "system audio" in the Record panel.

## Windows (NOT yet tested)
Run **First-time setup.bat**, install ffmpeg (`winget install Gyan.FFmpeg`), then
**Start Transcrb.bat**. Please report issues — this path needs a verification pass.
