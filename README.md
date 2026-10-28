# AnyLangMP42HunSync

[Magyar változat / Hungarian version](README.hu.md)

Hungarian dubbing for short videos (up to 10 minutes) with free, locally running AI. It runs on an Apple Silicon Mac with 16 GB RAM. The design is described in [DESIGN.md](DESIGN.md), which is in Hungarian.

The source language is currently English: the transcription step is set to `en`. Whisper itself can recognise many other languages.

## How it works

Pipeline:

1. **extract**: ffmpeg pulls the audio out of the video.
2. **separate**: Demucs splits the audio into speech and background (music, effects).
3. **transcribe**: Whisper (`mlx-whisper`, large-v3-turbo) transcribes the speech with word timestamps.
4. **sounds**: an AST sound-event classifier (AudioSet labels) finds animal sounds in the pauses between speech, so they are not lost from the dubbed track.
5. **voices**:
   - Speakers are separated by clustering XTTS speaker embeddings.
   - For every speaker the step:
     - builds a reference clip;
     - detects the voice type (male / female / child) from pitch;
     - measures the speaking tempo of the cloned voice;
     - corrects the pitch of cloned child voices.
6. **translate**: Gemma 3 12B (via Ollama) translates to spoken Hungarian.
   - Each line gets a syllable budget, so it fits the original timing.
   - The model is told each speaker's voice type, so children speak simply.
7. **synthesize**: XTTS-v2 speaks the Hungarian text in each original speaker's cloned voice.
   - Lines that are too long are shortened by the LLM in one batch.
   - The TTS model and the LLM are never in memory at the same time.
8. **align**: time stretching and pitch shifting place each line on the original timeline.
9. **mix**: the dub is mixed with the background and the kept animal sounds, then loudness-normalised.
10. **mux**: writes the output MP4 with a Hungarian track (default) plus the original track, an `.srt` subtitle and a report.

On the MacBook Air M5 (16 GB), a 9-minute cartoon with 5 speakers takes about 20 minutes end to end.

## Installation (Apple Silicon Mac)

```bash
brew install ffmpeg uv ollama
ollama serve &                 # or: brew services start ollama
ollama pull gemma3:12b
uv sync
```

The first run downloads these models:

| Model | Size |
|---|---|
| Whisper | ~1.6 GB |
| Demucs | ~80 MB |
| XTTS-v2 | ~1.9 GB |
| AST sound classifier | ~350 MB |

**License note:** XTTS-v2 is released under the Coqui Public Model License, which allows **non-commercial use only**.
- To accept the license, set `xtts_accept_license: true` in `config.yaml`.
- If you do not accept it, use `tts: edge`. This uses online Microsoft voices without cloning, chosen by voice type.

## Web UI

A local web interface in English and Hungarian (built with Gradio). It is reachable only from this machine.

```bash
uv run mp4translator ui          # opens http://127.0.0.1:7860
```

- **Dubbing:**
  - upload a video, which creates a job in `work/<name>/`;
  - start, stop and resume it, or rerun from a chosen step;
  - follow the step progress and the live log;
  - start Ollama with one click if it is not running.
- **Settings:** translation model (recommended list; ✓ marks downloaded models, and a button downloads the selected one with progress), Whisper and Demucs model, form of address, voice engine (with the XTTS license checkbox), background mode, speaker threshold, speed-up and pitch-correction limits, animal sounds and glossary. They are saved per job to `work/<name>/config.yaml`.
- **Result:** watch the dubbed video (the browser plays the Hungarian track), download the MP4, the subtitle and the report, and read the report.
- **Lines:** a table of every line (time, speaker, English, Hungarian, status).
  - Click a line to hear the original and the Hungarian audio, then fix the Hungarian text or the speaker. A line can also be regenerated with the same text when the voice got it wrong.
  - **Apply** regenerates only the changed lines (✏️), then rebuilds the video; one line takes about 30 seconds.
  - Hand-edited text is spoken as written: the LLM never shortens it, and a line that is too long is sped up instead.
  - Filters show only the lines that were sped up a lot or run into the next line (⚠️), or only the changed ones.
  - A single line's pitch can be raised or lowered (semitones, on top of the speaker's), with a preview before saving.
- **Speakers:**
  - see each speaker's detected voice type and pitch;
  - listen to the original sample and the cloned voice;
  - override the voice type and rerun from the voices step;
  - set the pitch of a speaker's cloned voice in semitones (with a preview), or go back to the automatic correction. This only remixes the video (about 10 seconds), no voice is regenerated, and the value is kept in the job's `config.yaml` (`speaker_pitch`);
  - preview the speaker count per clustering threshold.

The language follows the browser and can be switched at the top right. Only one job runs at a time; this keeps 16 GB of RAM enough. A job keeps running in the background even if the UI is closed, and the UI picks it up again when restarted. Jobs started from the command line also appear in the UI.

## Usage

```bash
uv run mp4translator dub video.mp4
```

Output:
- `video_hu.mp4`: track 1 is Hungarian, track 2 is the original audio;
- `video_hu.hu.srt`: Hungarian subtitles;
- `work/video/report.md`: speakers, voice types, kept sounds, and segments that needed speed-up or shortening.

Useful options:

- `--from-step <step>`: rerun from a step. Use it after editing `text_hu` or `speaker` in `work/video/manifest.json`, or after changing settings.
- `--llm`, `--whisper`, `--demucs`, `--tts`: choose the models for one run (see below).
- `-c config.yaml`: settings, including:
  - formal or informal address;
  - glossary;
  - speed-up limits;
  - speaker clustering sensitivity (`speaker_threshold`);
  - manual voice types (`speaker_kinds: {SPK2: child}`);
  - pitch correction (`pitch_correction: off | child | all`), or a manual pitch per speaker (`speaker_pitch: {SPK1: 2.5}`);
  - animal sound detection (`keep_animal_sounds`).

### Choosing models

`uv run mp4translator models` lists the recommended models and the Ollama models already downloaded. Set them in `config.yaml` (`llm`, `whisper_model`, `demucs_model`, `tts`), on the command line, or in the web UI.

| Purpose | Recommended | Alternatives |
|---|---|---|
| Translation (Ollama) | `ollama:gemma3:12b` | `gemma3:4b` (fast), `translategemma:12b` / `:4b` (translation-tuned), `gemma4:12b`, `gemma4:e4b`, `qwen3:8b` (experimental) |
| Speech recognition | `mlx-community/whisper-large-v3-turbo` | `-turbo-4bit` (smaller), `whisper-large-v3-mlx` (most accurate), `distil-whisper-large-v3` (English only), `whisper-small-mlx` |
| Separation | `htdemucs` | `htdemucs_ft` (cleaner voice, ~4x slower), `mdx_extra` |
| Voice | `xtts` | `edge` |

- Download an Ollama model with `ollama pull <name>` or the download button in the web UI. Whisper and Demucs models download automatically on first use.
- Models that "think" (qwen3, gemma4) get `think: false`, so they answer directly.
- Only Gemma 3 12B has been tested for Hungarian quality; listen to the result before switching.

### Inspecting speakers

```bash
uv run mp4translator speakers --workdir work/video
```

This prints how many speakers each clustering threshold would produce and can export preview clips.

Reference clips and calibration sentences for each speaker are saved in `work/video/speakers/`, so you can listen to them.

## Stop and resume

- **Stop:** press `Ctrl+C`, or run `uv run mp4translator stop` from another terminal. Models are released and the command to resume is printed.
- **Resume:** run the same `dub` command again.
  - Finished steps are skipped.
  - An interrupted step continues where it stopped: finished translation batches and synthesized segments are kept.
- **Status:** `uv run mp4translator status` shows step progress, segment counts and speaker voice types.
- Only one run can work on a work folder at a time (`work/<video>/run.pid`).

## iCloud

If the project lives in an iCloud-synced folder such as `Documents`, make `.venv`, `work` and `output` symlinks to `.venv.nosync`, `work.nosync` and `output.nosync`. iCloud does not sync `.nosync` folders. Otherwise it may evict files, and Python will no longer find the installed package.

## Tests

```bash
uv run --group dev pytest -q
```

## Further documents (Hungarian)

- [DESIGN.md](DESIGN.md): architecture and roadmap
- [OPTIMIZATION.md](OPTIMIZATION.md): speed measurements and optimizations
- [STATUS.md](STATUS.md): current state and next steps
