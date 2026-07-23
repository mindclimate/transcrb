# Transcrb — local, private, high-accuracy meeting transcription (EN + HE)

**Date:** 2026-07-23
**Location:** `/Users/aladdin/PROJECTS/Transcrb` (its own repo/project)
**Status:** design approved, ready for implementation plan

---

## 1. Purpose

A local program that transcribes meetings with the **highest achievable accuracy in English and Hebrew**, running **fully offline for privacy**. Output feeds the Work Brain's existing ingestion pipeline (optional, opt-in) and is usable standalone. Friendly enough for non-technical people to run on a machine where it's set up.

**Primary user:** Adin. **Secondary:** friends/family — initially Adin runs it for them; later they may run it on their own machines.

## 2. Non-goals (YAGNI)

- **Chat over the transcript** — explicitly excluded.
- **Standalone AI summary** — for Adin, the Work Brain already does summarization + decision/action-item extraction better. Deferred as a future pointer for the friends/family case (needs a local LLM).
- **Live/real-time transcription** — batch only for now. Documented as a future pointer.
- **Signed, zero-setup distributable app bundle** — deferred future pointer (large: packaging torch + models, per-arch builds, Apple notarization).
- **Native macOS ScreenCaptureKit audio capture** — deferred future pointer (Mac-only native Swift; breaks pure-Python).

## 3. Requirements (decisions made during brainstorming)

| Decision | Choice |
|---|---|
| Workflow | **Batch, post-meeting** (drop a file / record, transcribe after). |
| Languages | English + Hebrew. Usually one language per meeting; sometimes mixed (mixed is lower priority). **Hebrew accuracy is a hard requirement.** |
| Speaker labels | **Yes — diarization** (who said what). |
| Home | **Standalone project.** NOT folded into brain-kit (would bloat every brain clone with heavy ML deps). Brain coupling is a thin optional output adapter. |
| Reuse | Used across all Adin's projects; portable to friends/family. |
| Engine | **faster-whisper (CTranslate2)** + Hebrew-tuned model + **pyannote 3.1** diarization. |
| Language selection | **Auto-detect, override with `--lang`.** |
| Interface | **Local web GUI** (friendly, no terminal) on top of an engine core; CLI comes free from the same core. |
| Distribution | **Level A**: folder + double-click launcher + automated first-time setup. Level B (bundled/signed) deferred. |
| Platforms | **Mac verified now**; Windows structured + scripted but **marked untested** until tried on real hardware. |
| Recorder | **IN** — mic/in-person **and** system-audio via BlackHole (opt-in). |
| UI style | **Reuse the Workbench design system** (`.brain-kit/template/app.css` tokens + bevel + components) to match the brain dashboard and minimize frontend work. |

## 4. Environment constraints

- **Hardware:** Apple M1 Pro, 16GB RAM, arm64. Fits the chosen stack comfortably with int8 quantization.
- **Python 3.14 is too new** for the ML ecosystem (no stable torch/CTranslate2 wheels). Setup MUST create a dedicated **Python 3.11 virtualenv** (via `uv` or `pyenv`) — never touches system Python.
- No ffmpeg/whisper currently installed — clean slate; setup installs everything.

## 5. Architecture

```
Transcrb/
├── ▶ Start Transcrb.command       # macOS double-click launcher (bash)
├── ⚙ First-time setup.command     # macOS double-click setup (bash + uv)
├── Start Transcrb.bat             # Windows launcher (untested)
├── First-time setup.bat           # Windows setup (untested)
├── engine/                        # THE PRODUCT — pure Python, no UI knowledge
│   ├── audio.py                   # ffmpeg: any format -> 16kHz mono WAV
│   ├── detect.py                  # sample ~30s (multi-point) -> he/en language ID
│   ├── transcribe.py              # faster-whisper: load matching model, run
│   ├── diarize.py                 # pyannote 3.1: speaker turns
│   ├── merge.py                   # align transcript words <-> speaker turns
│   ├── record.py                  # mic + system-audio (BlackHole) capture
│   └── output.py                  # write md/srt/json (+ optional inbox copy)
├── cli.py                         # transcribe <file|folder> [--lang] [--no-diarize] [--inbox PATH]
├── web/
│   ├── server.py                  # small FastAPI/Flask server on localhost
│   └── index.html                 # drag-drop, record, language picker, progress, result
├── web/static/app.css             # Workbench design system (copied/derived from brain-kit template)
├── models/                        # downloaded weights (gitignored, multi-GB)
├── out/                           # default transcript destination
├── config.toml                    # defaults: output dir, optional inbox, HF token, quantization
├── setup.sh                       # shared setup logic (py3.11 venv, deps, models)
└── tests/
    └── fixtures/                  # tiny EN / HE / two-speaker clips
```

**Design principle:** `engine/` is the product. CLI and web are thin front-ends calling the same functions. Each engine module has one job and a clear input→output contract, independently testable.

## 6. Pipeline & models

**Step 1 — Normalize (ffmpeg).** Any input (`.m4a/.mp3/.wav/.mp4/…`) → 16kHz mono WAV.

**Step 2 — Language detect.** Sample ~30s from several points in the file (meetings open with small talk), run Whisper language ID → `he` or `en`. Overridable via `--lang`. Low confidence → fall back to config default and log it.

**Step 3 — Transcribe (faster-whisper / CTranslate2).**
- **English → `large-v3`** (CT2, `int8` / `int8_float16` quantization for 16GB).
- **Hebrew → the ivrit.ai fine-tune** (`ivrit-ai/whisper-large-v3` family), converted to CTranslate2 format at setup. **This is the single biggest Hebrew-accuracy lever.**
- Word-level timestamps ON (needed for diarization alignment).

**Step 4 — Diarize (pyannote 3.1), parallel.** Runs on the same WAV; acoustic, language-agnostic → works equally for Hebrew. Needs a one-time HuggingFace token + accepting the (free) model license; fully offline after download.

**Step 5 — Merge.** Align transcript words to speaker turns by timestamp overlap; "majority overlap wins" for words spanning a boundary. Handles overlapping speech gracefully.

**Step 6 — Output.** See §8.

*Hebrew + diarization:* transcript text comes from the Hebrew model (accurate), speaker labels from pyannote (language-agnostic). RTL preserved in output.

## 7. Recorder

A **Record** panel in the web UI:
- **Source picker:** Microphone (in-person) / System audio (remote call) / Both mixed.
- **System audio** via **BlackHole** virtual audio device. First-time setup offers guided install (Mac: brew). If absent, the option is disabled with a "set up" link — never crashes.
- **Flow:** Record → Stop → WAV drops into the same intake folder a manual drop uses → identical pipeline runs. The recorder is just another intake path, not a pipeline fork.
- Live elapsed timer + level meter. Audio written to disk continuously (crash-safe), not buffered in memory.

## 8. Outputs, config, error handling

**Outputs** → `out/<meeting-name>/` (all local):
- `transcript.md` — **primary**, human-readable, speaker-labeled, timestamped, RTL-preserved.
- `transcript.srt` — subtitles for replay.
- `transcript.json` — structured (segments, speakers, timings, detected language, model used) — the form the brain ingests.
- Optional copy into a `--inbox` target (WorkBrain or any project's `inbox/`).

**Config** (`config.toml`): default output dir, optional inbox path, HuggingFace token, default quantization, fallback language.

**Error handling:** every stage fails loud and specific — missing ffmpeg → "run setup"; missing HF token → exact steps; corrupt audio → names the file; OOM → suggest smaller quantization. **Diarization failure still yields the transcript** (degrade, don't lose work).

## 9. UI / design system reuse

Reuse the **Workbench design system** (`design/workbench-design-system.md` + `.brain-kit/template/app.css`): copy the `:root` token block, the `.raised`/`.inset` bevel motif, and the titlebar / statusbar / button / field / table components. Zero webfonts (system + monospace stacks). Result looks like a sibling of the brain dashboard; near-zero original frontend CSS. Self-contained (inlined/local assets) so the local server needs no internet.

## 10. Testing

- Engine modules unit-tested against tiny committed fixture clips (`tests/fixtures/`: short EN, HE, two-speaker).
- **Merge logic** (word↔speaker alignment) gets focused tests — the trickiest, most bug-prone unit.
- GUI + recorder are thin → manual test pass.
- Mac path fully verified. Windows launcher/setup scripts written but labeled "needs a test pass on real Windows hardware."

## 11. Brain integration

Transcrb stays a separate repo. The Work Brain **tracks** it as a project/workstream (a few markdown files) but does not host the code. The only runtime coupling: `--inbox /path/to/brain/inbox` copies a finished transcript into the brain's intake, where existing ingestion (summary, decisions, tasks) takes over — exactly like the current Vladi-call RTF pattern.

## 12. Future pointers (documented, NOT built now)

1. **Live/real-time mode** — streaming transcription during the meeting. Different architecture, lower accuracy; revisit if needed.
2. **Standalone local-LLM summary** (Ollama) — decisions/action items for the friends/family case (Adin uses the brain instead).
3. **Native macOS system-audio capture** (ScreenCaptureKit Swift helper) — no-install, Mac-only.
4. **Level B distribution** — signed, notarized, zero-setup double-click bundle with models baked in; per-arch builds.
