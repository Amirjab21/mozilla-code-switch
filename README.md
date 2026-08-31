# Repository directory

| Location | Description |
| --- | --- |
| [Miami audio explorer](analysis/audio_explorer.html) | A browser page for playing Bangor Miami Spanish–English corpus recordings alongside their transcript and metadata. |
| [Miami processing pipeline](processing/README.md) | Standalone scripts that create VAD-trimmed, normalized 16 kHz training clips and their transcript CSV. |
| [Processed audio analysis](analysis/processed-audio-analysis.html) | An interactive browser for reviewing VAD records, their original audio segments, transcripts, and waveforms. |
| [VAD comparison](analysis/vad-comparison.html) | Compare each stage-01 timestamp segment with its stage-02 VAD-trimmed output, duration retention, and waveforms. |
| [Whisper evaluation](analysis/whisper-evaluation.html) | A sortable review page for Whisper transcripts, WER, language-group WER, and aligned differences. |
| [Dataset configuration](dataset.json) | Defines the corpus folders and recording list used by both reusable analysis pages. |

## Top-level commands

Run these commands from the repository root. Install the processing environment first:

```bash
uv sync --project processing
```

Start the local server for the analysis pages and on-demand Hugging Face Whisper transcription. It includes byte-range support, so the long source MP3s can seek immediately when you click a transcript line:

```bash
uv run --project processing python processing/06_huggingface_whisper_server.py
```

Then open `http://localhost:8000/analysis/audio_explorer.html` or `http://localhost:8000/analysis/training-run-evaluation.html`.

Run full training with either named configuration:

```bash
uv run --project processing python processing/05_train.py \
  --config processing/runs/full_training.yaml

uv run --project processing python processing/05_train.py \
  --config processing/runs/full_training_lora.yaml
```

Online W&B runs require `WANDB_API_KEY` in the root `.env`. See [the processing guide](processing/README.md) for data preparation, preview runs, and individual stage commands.
