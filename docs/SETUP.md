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

## Recording
The Record panel lists your input devices; pick one and hit Record. The clip is
saved under `out/recordings/` and transcribed automatically when you stop.

**Grant microphone access first.** macOS asks the app that launched the server —
Terminal — for permission the first time. If recordings come out near-silent or
much shorter than expected, enable it under
System Settings → Privacy & Security → Microphone, then restart Transcrb.
Allow a second or two of lead-in: ffmpeg takes a moment to open the device.

## System audio (both sides of a call)
Pick **"Call audio + my mic (automatic)"** in the Record panel. That is the whole
setup. It captures the remote side *and* your own voice, and there is nothing to
route or switch.

It works by creating a CoreAudio **process tap** (macOS 14.2+) for the duration of
the recording, combined with your microphone in a temporary aggregate device. A tap
captures what macOS *plays*, not what a particular device receives, so:

- headphones, speakers, AirPods, an external interface — all identical;
- plugging or unplugging headphones **mid-call** changes nothing (verified: the
  system output was switched to a different device during a live recording and
  capture continued uninterrupted);
- Slack's own output picker does not matter;
- your system output is never modified, and you hear the call normally.

The tap and its device exist only while recording and are destroyed on stop, so
they never appear in other apps' device lists. `setup.sh` verifies at install time
that a tap can be created. If that check fails, the usual cause is
System Settings → Privacy & Security → **Screen & System Audio Recording**.

### Older macOS (before 14.2): the BlackHole fallback
Taps do not exist before macOS 14.2, so `setup.sh` creates two CoreAudio devices
instead — the ones you would otherwise build by hand in Audio MIDI Setup:

- **Transcrb Output (hear + capture)** — a Multi-Output Device of your output plus
  BlackHole. Select it as the *system output* during a call, so you hear the call
  and BlackHole receives a copy.
- **Transcrb Input (call + mic)** — an Aggregate Device of BlackHole plus your
  microphone. Select it as the *Input* in the Record panel, so the recording has
  both the remote side and your own voice.

Re-run `.venv/bin/python -m engine.macos_audio` after switching headphones to
rebuild them around the new device. BlackHole itself only appears **after a
reboot** following install.

On this path the gotchas are real: plugging headphones in or out resets the system
output away from the Multi-Output Device, and a wrong output means BlackHole gets
nothing. **BlackHole is a virtual output, not a microphone** — selecting it while
the call plays to your headphones records digital silence: a full-size file, no
error, and a transcript that is the word "you" once every 30 seconds, which is what
Whisper emits for silence.

### The silence guards
Whichever path you use, the Record panel probes the input before it starts and
refuses to record a silent one, shows a live level meter while recording and warns
after 10 seconds of no signal, and will not spend 40 minutes transcribing a
recording that turned out to be silent. If the meter reads "no signal", fix it
before the meeting rather than discovering it afterwards.

## Windows (NOT yet tested)
Run **First-time setup.bat**, install ffmpeg (`winget install Gyan.FFmpeg`), then
**Start Transcrb.bat**. Please report issues — this path needs a verification pass.
