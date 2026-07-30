# Transcrb

Meeting transcription that runs entirely on your own Mac. Audio never leaves the
machine — no account, no upload, no cloud service. English and Hebrew, including
calls that switch between them mid-conversation, with speaker labels.

## Install (once)

1. Double-click **First-time setup.command**.
   If macOS says "unidentified developer", right-click it → **Open** instead.
2. Leave it running. It installs what it needs and downloads the speech models
   (a few GB, one time only). This takes a while on the first run.
3. When it says setup is complete, close the window.

## Use it

Double-click **Start Transcrb.command**. Your browser opens with two panels:

- **Transcribe a file** — drop in any audio or video file.
- **Record** — record a call. Leave the input on
  *"Call audio + my mic (automatic)"* and it captures both sides of the
  conversation, with nothing to configure. Recordings transcribe themselves when
  you press stop.

Transcripts are saved in `out/`, one folder per recording, as Markdown, JSON and
subtitles. Close the terminal window to stop Transcrb.

Transcription is faster than real time: roughly 9 minutes of processing for a
25-minute meeting, measured on an M1 Pro. Speech recognition runs on the CPU and
speaker labelling on the Mac's GPU.

## Speaker labels

Naming who spoke needs a free HuggingFace token. Without one you still get a full
transcript, just with everyone as `SPEAKER_00`. See step 2 of
[docs/SETUP.md](docs/SETUP.md) for the three-minute setup.

## What's in here

| | |
|---|---|
| `config/` | your settings — `config.toml` (yours, private) and an example to copy |
| `engine/` | the transcription pipeline |
| `web/` | the local web interface |
| `scripts/` | setup script called by the installer |
| `docs/` | [SETUP.md](docs/SETUP.md) — full setup, troubleshooting, how recording works |
| `out/` | your transcripts and recordings (created on first use) |
| `models/` | downloaded speech models (created by setup) |

## Windows

Run **First-time setup.bat**, then **Start Transcrb.bat**. This path is not yet
tested — please report what breaks.
