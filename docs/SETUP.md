# Setup

## macOS (verified)
1. Double-click **First-time setup.command**. If macOS warns "unidentified developer",
   right-click → Open. It installs ffmpeg + BlackHole, builds the Python 3.11 env,
   and downloads/converts the models (a few GB, one time).
2. Speaker labels need a free HuggingFace token:
   - Create one at https://huggingface.co/settings/tokens
   - Put it in `config/config.toml` as `hf_token = "hf_..."` (copy from `config/config.example.toml`),
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
     _default_factory(tomllib.load(open('config/config.toml','rb'))['hf_token']); print('diarization OK')"
     ```
3. Double-click **Start Transcrb.command** — your browser opens the UI.

## Speed
Faster than real time: **about 9 minutes for a 25-minute meeting** with speaker
labels on, measured end-to-end on an M1 Pro with a single-language recording.
Picking the language explicitly instead of Auto-detect skips the language scan and
saves a further ~20%. A meeting that switches language mid-call loads both models
and transcribes span by span, so expect somewhat longer.

Speech recognition is CPU-only — CTranslate2 has no Metal backend — but speaker
labelling is not, and both were re-measured rather than assumed. For 4 minutes of
audio, before and after:

| stage | was | now | |
|---|---|---|---|
| language scan | 24.9s | 19.2s | unchanged code |
| transcription | 267.7s | 48.0s | **5.6x** — turbo models |
| speaker labels | 244.5s | 18.4s | **13.3x** — pyannote on Metal |
| **total** | **537.1s** | **85.7s** | **6.3x** |

Those figures come from a 4-minute clip, so they were re-checked on a real
26-minute meeting (English picked explicitly, 3 speakers) to confirm the numbers
hold at meeting length rather than only extrapolating:

| stage | 25.9 min of audio | rate |
|---|---|---|
| convert audio | 0.3s | — |
| transcription | 341.8s | 4.5x realtime |
| speaker labels | 92.0s | 16.9x realtime |
| **total** | **~7.3 min** | **3.5x realtime** |

Speaker labelling gets *cheaper* per minute as recordings grow (4.6s per
audio-minute at 4 minutes, 3.6s at 26), so the short-clip measurement was
conservative. Transcription is the flat cost and dominates at any length.

Two things produced that, and the order is worth knowing if you tune further:

1. **Speaker labelling was nearly half the runtime** and nobody had noticed,
   because it looked like a fixed cost. pyannote moves to the GPU with one call;
   an op MPS cannot handle falls back to the CPU rather than dropping labels.
2. **Model architecture beat compute backend.** The turbo models keep all 32
   encoder layers but have 4 decoder layers instead of 32. Turbo on the CPU
   (4.25x realtime) measured *faster* than the previous weights on the GPU via
   MLX (3.48x) — so the cheap change was swapping models, not rewriting the
   inference backend.

Two further options were measured and **one was rejected**:

- **Batched decoding** is a further 1.7x and was turned down: it drops speech.
  On four minutes of English it returned 596 words where sequential returned 668,
  and on a Hebrew clip it silently lost an entire opening sentence. A transcript
  that is missing words while reading fluently is the failure this project cares
  most about avoiding, so the speed is not worth it.
- **Whisper on Metal via MLX** is genuinely promising and untaken: a turbo model
  reached 11.8x realtime against the 4.25x we now get on the CPU. It needs the
  ivrit.ai turbo weights converted to MLX format and a second inference backend
  to maintain, so it is a project rather than a switch.

The next cheap win is the language scan, now ~20% of the total. Probing at a 4s
hop and refining only around disagreements would roughly halve it while still
catching every span longer than the 5s minimum.

## Languages, including meetings that switch mid-call
Leave the picker on **Auto-detect** and a recording that moves between Hebrew and
English is handled correctly: each stretch is transcribed by the model that
matches it — Hebrew by the ivrit.ai model, English by large-v3-turbo — and the
transcript reports `he+en` with each line tagged by language.

This matters more than it sounds. Whisper decides one language per file from its
opening 30 seconds. Given the wrong one it does not fail or skip — it
**translates**, so a Hebrew answer in an English-labelled call comes back as
fluent English nobody actually said, with nothing in the output to reveal it.
Auto-detect therefore scans the recording for language spans before
transcribing, rather than labelling the whole file from its first half minute.

How it works, and why it looks the way it does. Two measured facts drove it:

1. **A window's reported language reflects its opening, not its majority — at
   full confidence.** A window that was 74% Hebrew came back `en (0.96)`. So
   there is no clever signal for "this window contains a switch"; the only thing
   that bounds the error is probing often enough. The scan therefore probes
   uniformly every 2 seconds, each probe reading 6 seconds for context but
   labelling only the 2 seconds at its start.
2. **A small model is both cheaper and better at this.** `base` scored 19/19 on a
   clip that large-v3 read as pure English, at ~20x less cost per probe. So
   language ID uses `base` (downloaded by setup, ~150MB) while transcription
   still uses the full-size turbo models. A Hebrew-only meeting never loads the
   English model at all.

Contiguous probes become spans; anything under 5 seconds is treated as detector
noise and absorbed into its neighbour, and an unsupported answer (Whisper
occasionally says Arabic for Hebrew) inherits from its neighbours.

Costs and limits:
- The scan runs at about 0.08x realtime — roughly 2 minutes for a 24-minute
  meeting. That was noise next to transcription before; now that transcription is
  5.6x faster the scan is about a fifth of the total, so it is the next thing
  worth optimizing. Models load in ~4s each.
- Boundary precision is the 2-second probe hop. Switching language
  **mid-sentence** can leave a second or two on the wrong side of a boundary,
  and that fragment gets translated. Switches at sentence boundaries — how
  people actually speak — are handled cleanly.
- Accuracy is verified against synthesized bilingual speech
  (`tests/fixtures/make_mixed_speech.sh`), which is cleaner than a real meeting.
  Treat the 19/19 as a floor-setting sanity check, not a field measurement.
- **Hebrew only** / **English only** skip the scan entirely. If you pick one and
  the meeting turns out to be mixed, you are back to the translation behaviour
  above, so prefer Auto-detect unless you are certain.

## Recording
The Record panel lists your input devices; pick one and hit Record. The clip is
saved under `out/recordings/` and transcribed automatically when you stop.

**Grant microphone access first.** macOS asks the app that launched the server —
Terminal — for permission the first time. If recordings come out near-silent or
much shorter than expected, enable it under
System Settings → Privacy & Security → Microphone, then restart Transcrb.
Allow a second or two of lead-in: opening the device takes a moment.

### What does the recording
`native/transcrb-capture`, built from `native/capture.swift` by
`scripts/build-native.sh` during setup. It exists because ffmpeg's macOS audio
input cannot be made reliable: it holds one pending buffer and blocks the
capture callback until that buffer is read, giving the capture about 10ms of
tolerance before CoreAudio starts discarding a meeting, with no error and no gap
in the file. That cost two recorded meetings in August 2026, which came back as
"Thank you." and "I don't know." on a loop.

The native recorder keeps 30 seconds of ring buffer, writes from a separate
thread, and counts the frames CoreAudio never handed it. Measured under
`taskpolicy -b`, the same throttle that caused the original failure: ffmpeg lost
88.7% of a 30-second capture, the native recorder lost 0.1%.

ffmpeg still records if the binary has not been built, and the Record panel says
so at the start of the meeting rather than at the end. Build it with
`bash scripts/build-native.sh`; it needs the Xcode Command Line Tools
(`xcode-select --install`). `run.sh` rebuilds it on service start whenever
`capture.swift` has changed.

The level check before recording goes through whichever recorder is going to do
the meeting, on purpose. Probing with one binary and recording with another
would let a microphone permission granted to the first and refused to the second
pass the check and then record an hour of silence.

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
they never appear in other apps' device lists. `scripts/setup.sh` verifies at install time
that a tap can be created. If that check fails, the usual cause is
System Settings → Privacy & Security → **Screen & System Audio Recording**.

### Older macOS (before 14.2): the BlackHole fallback
Taps do not exist before macOS 14.2, so `scripts/setup.sh` creates two CoreAudio devices
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

### The guards
Whichever path you use, the Record panel probes the input before it starts and
refuses to record a silent one, shows a live level meter while recording and warns
after 10 seconds of no signal, and will not spend 40 minutes transcribing a
recording that turned out to be silent. If the meter reads "no signal", fix it
before the meeting rather than discovering it afterwards.

A loud recording is not a complete one, so completeness is watched separately.
The timer reads `● recording 12:34 · 100% captured`, updated every second from
the recorder's own frame count, and drops below 100% the moment audio is being
lost. Stopping reports the same figure. Below 65% it refuses to transcribe and
offers "Transcribe anyway" instead, because a recording missing most of itself
produces a page of one repeated phrase and hides the real problem.

### Recordings the app never finished
Closing the window, a restart, or reinstalling the service mid-meeting all end a
recording without anyone stopping it. The audio survives, because the recorder
writes to disk as it goes, but nothing pointed at it: one 93MB recording sat
unnoticed in `out/recordings/` for two days. Those are now listed in the Record
panel with a Transcribe button, and their wav headers are repaired as they are
listed. Until that repair they claim to hold zero samples, so QuickTime, Finder
and libsndfile all read an hour of meeting as an empty file.

## When something goes wrong
Transcrb keeps a log at **`logs/transcrb.log`**, so a failure is still
diagnosable after the terminal window is closed. It holds the full traceback of
anything that failed, plus the warnings that are easy to miss in passing — most
usefully, notice that speaker labelling could not use the GPU and fell back to
the CPU, which is roughly four times slower and otherwise looks identical to a
normal run. The file is capped at 2MB with three older copies kept, so it cannot
grow without bound. The terminal still shows everything it always showed.

If a transcript comes back wrong, that log and the recording under
`out/recordings/` are the two things worth keeping.

### One transcription at a time
The server runs a single transcription at a time and a second request waits its
turn, reporting "Waiting for the current transcription to finish…". Running two
at once is *slower than running them in sequence* — each loads its own multi-GB
model, holds the whole recording in memory, and competes for the same GPU — so
queueing is the faster behaviour as well as the safer one.

### Re-transcribing keeps the older result
Running a recording again — a different speaker count, the language picked by
hand — no longer destroys the first transcript. The earlier files move to
`out/<name>/previous/<when-they-were-made>/`, so the two can be compared:

```
out/recording-20260730-124519/
  transcript.md                     <- newest run, always
  previous/20260730-134728/         <- the run made at 13:47
  previous/20260730-143255/         <- the run made at 14:32
```

`transcript.md` and the other canonical names always hold the newest run, so
nothing reading them needs to know this exists. Each kept run costs the size of
its transcripts — a few hundred KB for a meeting, mostly `words.json` — so if
you have re-run something many times, `previous/` is safe to delete.

The copy in your inbox is **replaced** rather than kept, deliberately: it is a
drop point that gets consumed, and duplicates there would be ingested twice.

## Windows (NOT yet tested)
Run **First-time setup.bat**, install ffmpeg (`winget install Gyan.FFmpeg`), then
**Start Transcrb.bat**. Please report issues — this path needs a verification pass.
