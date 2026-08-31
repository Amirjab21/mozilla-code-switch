# Miami corpus processing pipeline

Produces training-ready WAV clips and `train.csv` from the raw Miami corpus MP3 and CHAT files. The pipeline is deliberately split into three independently runnable stages so that a completed stage does not need to be repeated after a later failure.

Ensure `uv` and `ffmpeg` are installed, then create the project environment and run the complete pipeline:

```bash
uv sync --project processing
uv run --project processing python processing/00_run_pipeline.py
```

## Processing once, named training runs

Run processing once into the shared root `processed/` directory. This produces `processed/01_segments.csv`, `processed/02_vad.csv`, and the labelled `processed/train.csv` that every later training run reuses:

```bash
uv run --project processing python processing/00_run_pipeline.py \
  --output-dir processed \
  --skip-inference
```

Named YAML files in `processing/runs/` are training-only configurations. They set the W&B run name and reference the shared training manifest, while storing checkpoints, split manifests, and evaluation CSVs in their own run-specific output directories.

For the full named training run, use:

```bash
uv run --project processing python processing/05_train.py \
  --config processing/runs/full_training.yaml
```

`processing/runs/training_preview.yaml` provides the bounded smoke-test equivalent for an already-created preview manifest. Command-line options still override YAML values when needed.

## Inputs and outputs

- Raw audio: `miami/audios/*.mp3`
- CHAT transcripts: `miami/chat/*.cha`
- Word-level annotations: `miami/word_level_tsvs/*_cgwords.tsv`
- Stage 1 output: timestamp-aligned source clips in `processed/01_segments/` and `processed/01_segments.csv`
- Stage 2 output: VAD-trimmed clips in `processed/02_vad/` and `processed/02_vad.csv`
- Stage 3 output: normalized training clips in `processed/audio/` and `processed/train.csv`
- Stage 4 output: Whisper transcription evaluation in `processed/04_whisper_evaluation.csv`
- Stage 5 output: token-language fine-tuning artifacts in `processed/05_train/`

The final `processed/train.csv` contains `audio_path`, `transcript`, `word_langids`, and `language_counts`. `word_langids` is a JSON array of `{ "word", "langid" }` objects, in the same order as the whitespace-separated transcript; `language_counts` is a JSON object that summarizes the retained labels. Intermediate manifests also retain `source_audio`, `start_ms`, `end_ms`, `speakers`, and `utterance_ids` for traceability.

## Stage 1 — CHAT timestamp segmentation

`01_split_chat_segments.py` pairs every MP3 with its same-named `.cha` file and `_cgwords.tsv` annotation. It counts every CHAT speaker tier so its original utterance number aligns with the TSV `utterance_id`, then uses CHAT timing markers (`\x15start_end\x15`, measured in milliseconds) to place segment boundaries. Consecutive timestamped utterances are combined until adding the next one would exceed 20 seconds; a segment is therefore never longer than 20 seconds, though it can be shorter at a natural transcript boundary.

The TSV surface forms, rather than the CHAT display text, become the canonical training transcript. Parenthesised material and all punctuation are removed; angle brackets are removed while retaining the enclosed words; underscore-linked forms become separate words; and `www`/`xxx` placeholders are excluded. Every retained word keeps its TSV `langid`, including `eng`, `spa`, ambiguous (`eng&spa`), and mixed-morpheme (`eng+spa` / `spa+eng`) labels. `ffmpeg` extracts each time range as a mono 16-bit PCM WAV.

Key options: `--max-seconds` (default `20`), `--audio-dir`, `--chat-dir`, `--tsv-dir`, `--output-dir`, and `--manifest`.

## Stage 2 — Silero voice activity detection

`02_vad_trim.py` reads `01_segments.csv`, decodes each source WAV to mono 16 kHz PCM through `ffmpeg`, and passes the samples to Silero VAD. A speech probability threshold of `0.5` is used by default. Adjacent speech regions separated by 20 ms or less are merged, then the retained regions are concatenated to remove non-speech audio. The resulting clips are written as mono, signed 16-bit PCM WAVs at 16 kHz.

Rows containing no detected speech are kept in `02_vad.csv` with `filtered_out=True`; rows with speech are written to `02_vad/` and marked `filtered_out=False`. This makes rejected material inspectable without retaining a VAD output clip. The script decodes with `ffmpeg` rather than TorchCodec to avoid macOS audio-library compatibility issues.

Key options: `--threshold` (default `0.5`), `--merge-gap-ms` (default `20`), `--manifest`, `--output-dir`, and `--output-manifest`.

## Stage 3 — final normalization

`03_normalize_audio.py` processes only rows that were not filtered out by VAD. It forces mono output, resamples to 16 kHz, converts to signed 16-bit PCM WAV, and applies EBU R128 loudness normalization with target integrated loudness `-23 LUFS`, true peak `-2 dBTP`, and loudness range `7 LU` (`loudnorm=I=-23:TP=-2:LRA=7`). It preserves the token-level language fields in the final training manifest.

Key options: `--manifest`, `--output-dir`, and `--output-csv`.

## Stage 4 — Whisper language inference

`04_whisper_evaluation.py` runs OpenAI Whisper over VAD-retained audio, stores the returned transcription, and calculates standard word error rate (WER) against the TSV-derived reference transcript. Its alignment records correct words, substitutions, deletions, and insertions. It also groups those counts and WER by the raw TSV `langid`, preserving ambiguous labels such as `eng&spa` rather than treating them as bilingual.

The model is selected through `--model` (default `medium`), so the inference stage can be rerun with another multilingual Whisper checkpoint without changing code. Use `--device auto` by default or choose `cpu`, `cuda`, or `mps`. Whisper downloads model weights into `models/whisper/` on its first run.

For the preview set only:

```bash
uv sync --project processing
uv run --project processing python processing/04_whisper_evaluation.py \
  --manifest processed/preview/02_vad.csv \
  --output-csv processed/preview/04_whisper_evaluation.csv \
  --model medium
```

## Running individual stages

Each stage accepts `--help` and can be run independently. For example:

```bash
uv run --project processing python processing/01_split_chat_segments.py --help
```

## Stage 5 — train the local Whisper token-language fork

`05_train.py` fine-tunes `models/whisper_lid`, which preserves Whisper Small's normal ASR head and adds a parallel token-level language classifier. It requires the final manifest to include `word_langids`; if `processed/train.csv` has only `audio_path` and `transcript`, re-run stages 01–03 first to regenerate the labelled outputs.

The script makes a deterministic 90/10 train/test split with seed `1337`. It holds entire recordings out when more than one recording is present, preventing adjacent clips from the same source recording appearing in both partitions; a one-recording preview falls back to a clip-level split. It writes both split manifests to `processed/05_train/`.

At every 250 training steps by default, it evaluates on a fixed held-out subset of up to 100 clips. The evaluation logs combined `eval/loss`, separate token and language-ID losses, token-language accuracy, and greedy-decoding WER to Weights & Biases. These constants are defined at the top of `05_train.py` as `EVAL_EVERY_STEPS` and `EVAL_SET_SIZE`; command-line options can override them for one run.

`best_eval_loss` starts at infinity and is updated whenever a lower held-out `eval/loss` is observed. The corresponding compact PEFT adapter is saved as `best_lora/`. Each evaluation overwrites `eval_predictions.csv`; the best result is preserved as `best_eval_predictions.csv`. Both CSVs contain `audio_file_path`, the ground-truth transcript and language labels, predicted transcript and word-level predicted language labels, plus clip-level `wer`, `loss`, `token_loss`, and `language_id_loss`.

At completion, the script writes a self-contained audio review page for the best evaluation result to `analysis/training_run_eval/{datetime}_run_{runname}.html`. It has previous/next controls, an audio player for every held-out clip, and sorts by WER, combined loss, token loss, or language-ID loss. W&B receives only scalar training/evaluation metrics; audio is never uploaded. The local evaluation CSV preserves the audio paths used by the local HTML review page.

For a reusable CSV viewer, open `analysis/training-run-evaluation.html` through a local web server. It defaults to the training preview CSV and can be pointed at another output using its inputs or a URL such as `analysis/training-run-evaluation.html?csv=../processed/05_train/eval_predictions.csv&audio_base=..`.

### On-demand Hugging Face Whisper transcription

The reusable evaluation viewer can also transcribe its currently selected audio clip with Hugging Face Whisper Small or Medium. Start the local transcription server instead of the basic Python HTTP server:

```bash
uv run --project processing python processing/06_huggingface_whisper_server.py
```

Open the viewer at `http://localhost:8000/analysis/training-run-evaluation.html`, select **Small** (the default) or **Medium**, then choose **Whisper transcription**. The server reads only repository-local audio paths and returns the transcript to the page; it does not upload the audio. It also supports HTTP byte-range requests, which lets the long source MP3s in `audio_explorer.html` seek immediately from a CHAT transcript timestamp. The first use of either variant downloads its Hugging Face checkpoint if necessary.

Add your W&B key to a root `.env` file (which is ignored by Git):

```text
WANDB_API_KEY=...
```

Then run:

```bash
uv sync --project processing
uv run --project processing python processing/05_train.py \
  --manifest processed/train.csv \
  --base-model small \
  --wandb-project miami-whisper-token-lid
```

For a no-upload smoke test, pass `--wandb-mode disabled`. Training saves `last_lora/` after each epoch and `final_lora/` at completion; these are compact PEFT adapters containing the LoRA weights and the language-ID head, not duplicate Whisper base weights.

### Limited training preview

Use `--max-samples` to select a deterministic subset before the 90/10 split and `--max-train-steps` to bound runtime. This still performs the complete train/test split, W&B logging, periodic held-out evaluation, and checkpoint writing:

```bash
uv run --project processing python processing/05_train.py \
  --manifest processed/preview/train.csv \
  --output-dir processed/preview/05_train \
  --max-samples 40 \
  --max-train-steps 10 \
  --epochs 10 \
  --batch-size 2 \
  --eval-every-steps 5 \
  --eval-set-size 10 \
  --wandb-mode disabled
```

### Limited end-to-end preview

To test the complete workflow from raw Miami recordings through the custom-model training script, first create a small labelled processing output. `--skip-inference` omits the optional Whisper Medium baseline evaluation in stage 4, keeping this preview focused on the training path:

```bash
uv run --project processing python processing/00_run_pipeline.py \
  --output-dir processed/training_preview \
  --max-recordings 2 \
  --max-segments-per-recording 25 \
  --skip-inference
```

Then train on at most 40 of those clips, for at most 10 optimisation steps:

```bash
uv run --project processing python processing/05_train.py \
  --manifest processed/training_preview/train.csv \
  --output-dir processed/training_preview/05_train \
  --max-samples 40 \
  --max-train-steps 10 \
  --epochs 10 \
  --batch-size 2 \
  --eval-every-steps 5 \
  --eval-set-size 10 \
  --wandb-mode disabled
```

This creates the deterministic 90/10 split, evaluates at steps 5 and 10, writes WER and language-ID metrics, and saves `last_lora/` and `final_lora/`. Once the local smoke test succeeds, remove `--wandb-mode disabled` to log online using `WANDB_API_KEY` from `.env`.

## Preview run

Use a separate output directory with `--max-recordings` and `--max-segments-per-recording` to create a small end-to-end preview without changing the full dataset. This example processes the first recording alphabetically and its first five segments:

```bash
uv run --project processing python processing/00_run_pipeline.py \
  --output-dir processed/preview \
  --max-recordings 1 \
  --max-segments-per-recording 5 \
  --inference-model medium
```

The preview training manifest is `processed/preview/train.csv`. The pipeline also writes `processed/preview/dataset.json`, ready for the processed-audio review page. Serve the repository root and open `analysis/processed-audio-analysis.html?config=../processed/preview/dataset.json`. Remove `processed/preview/` when it is no longer needed.

If an earlier run stopped during VAD, rerun only the remaining stages with the existing segment manifest:

```bash
uv sync --project processing
uv run --project processing python processing/02_vad_trim.py
uv run --project processing python processing/03_normalize_audio.py
```

## Training-audio duration distribution

`07_audio_duration_distribution.py` reads the `audio_path` entries in a training manifest and writes an SVG histogram whose bars are the probability of a clip falling in each duration bin. It uses the standard-library WAV reader, so it adds no dependencies:

```bash
uv run --project processing python processing/07_audio_duration_distribution.py \
  --manifest processed/train.csv \
  --output analysis/train_audio_duration_distribution.svg
```

Use `--bins 20` (or another positive number) to change the histogram resolution.
