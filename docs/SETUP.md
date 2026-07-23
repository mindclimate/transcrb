# Setup

## macOS (verified)
1. Double-click **First-time setup.command**. If macOS warns "unidentified developer",
   right-click → Open. It installs ffmpeg + BlackHole, builds the Python 3.11 env,
   and downloads/converts the models (a few GB, one time).
2. Speaker labels need a free HuggingFace token:
   - Create one at https://huggingface.co/settings/tokens
   - Put it in `config.toml` as `hf_token = "hf_..."` (copy from `config.example.toml`),
     or export `HF_TOKEN`.
   - **Accept the conditions on all three gated repos** while logged in. pyannote 4.x
     redirects internally to `community-1`, so accepting only the first two is not
     enough — diarization then fails silently and every line is labelled SPEAKER_00:
     - https://huggingface.co/pyannote/speaker-diarization-community-1
     - https://huggingface.co/pyannote/speaker-diarization-3.1
     - https://huggingface.co/pyannote/segmentation-3.0
   - Verify before a long run:
     ```
     .venv/bin/python -c "import tomllib; from engine.diarize import _default_factory; \
     _default_factory(tomllib.load(open('config.toml','rb'))['hf_token']); print('diarization OK')"
     ```
3. Double-click **Start Transcrb.command** — your browser opens the UI.

## Speed
Transcription is CPU-only (CTranslate2 has no Metal backend). On an M1 Pro expect
roughly **30–60 minutes for a 42-minute meeting** with speaker labels on. Picking the
language explicitly instead of Auto-detect skips a detection pass and one model load —
noticeably faster, and it guarantees Hebrew uses the ivrit.ai model.

## System audio (both sides of a call)
After BlackHole is installed, create a Multi-Output Device (Audio MIDI Setup) that
includes BlackHole, route your call audio to it, and pick "system audio" in the Record panel.

## Windows (NOT yet tested)
Run **First-time setup.bat**, install ffmpeg (`winget install Gyan.FFmpeg`), then
**Start Transcrb.bat**. Please report issues — this path needs a verification pass.
