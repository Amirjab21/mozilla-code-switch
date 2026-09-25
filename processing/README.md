# Indonesian and Javanese corpus processing pipeline

Produces 16 kHz WAV clips and a start-corrected transcript manifest from the Jember timestamped recordings and Indonesian development clips. All outputs are isolated in `processed_indonesia/`; generated analysis artifacts belong in `analysis_indonesia/`.

Ensure `uv` and `ffmpeg` are installed, then create the project environment and run the complete pipeline:

```bash
uv sync --project processing
uv run --project processing python processing/00_run_pipeline.py
```

Jember transcript starts are matched using Wav2Vec2 in `02_match_predicted_clip_start.py`. Before that stage writes its manifest, it applies the shared `text_normalisation.normalize_text` rules to every transcript from every dataset. It does not alter the Miami `processed/` directory.

The pipeline then adds room-noise copies, followed by combined speed, pitch,
random-amplitude, time-dropout, and room-reverb copies, before training. The
main processing outputs are:

- `processed_indonesia/01_segments.csv`
- `processed_indonesia/02_processed.csv`
- `processed_indonesia/03_dataset_with_augmented.csv`
- `processed_indonesia/03_1_dataset_with_perturbations.csv`
- `processed_indonesia/clip_start_matches.json`

## Processing once, named training runs

Run processing once into `processed_indonesia/`. This produces `processed_indonesia/01_segments.csv` and the corrected `processed_indonesia/02_processed.csv`:

```bash
uv run --project processing python processing/00_run_pipeline.py \
  --output-dir processed_indonesia
```

Named YAML files in `processing/runs/` are training-only configurations. They set the W&B run name and reference the shared training manifest, while storing checkpoints, split manifests, and evaluation CSVs in their own run-specific output directories.

For the full named training run, use:

```bash
uv run --project processing python processing/05_train.py \
  --config processing/runs/full_training.yaml
```

`processing/runs/training_preview.yaml` provides the bounded smoke-test equivalent for an already-created preview manifest. Command-line options still override YAML values when needed.

## Inputs and outputs

- Raw Jember recordings: `indonesian_data/Jember Javanese Spontaneous Speech Corpus/mp3 audio/*.mp3`
- Jember timestamps/transcripts: `indonesian_data/Jember Javanese Spontaneous Speech Corpus/Jember Javanese Spontaneous Speech Corpus - 1-200.tsv`
- Development clips and metadata: `indonesian_data/indonesian_dev/clips/*.mp3` and `indonesian_data/indonesian_dev/metadata.tsv`
- Common Voice evaluation clips: `indonesian_data/cv_indonesian/id/test.tsv` and `indonesian_data/cv_javanese/ss-corpus-jv.tsv`
- Stage 1 output: source-aligned WAV clips in `processed_indonesia/01_segments/` and `processed_indonesia/01_segments.csv`
- Stage 2 output: start-corrected Jember rows and all other dataset rows, with every transcript normalized using `text_normalisation.normalize_text`, in `processed_indonesia/02_processed.csv`; detailed matching results are stored in `processed_indonesia/clip_start_matches.json`
- Stage 3 output: original rows plus configured room-noise copies in `processed_indonesia/03_dataset_with_augmented.csv`
- Stage 3.1 output: all stage-3 rows plus combined augmentation copies in `processed_indonesia/03_1_dataset_with_perturbations.csv`. The `speed_pitch` YAML section controls copies, speed range, random-amplitude gain, RIR directory, reverb wet-mix range, time-dropout probability/chunk sizes/counts, and random seed. Defaults are `0.7–1.3×`, `-12–+6 dB`, `15–55%` wet reverb, and a 50% chance of dropping 1–3 chunks of 50–200 ms. RIR convolution uses one randomly selected channel and a soft limiter prevents clipping.

The VAD, audio-normalization, and evaluation scripts remain available as standalone utilities.

The final `processed/train.csv` contains `audio_path`, `transcript`, `word_langids`, and `language_counts`. `word_langids` is a JSON array of `{ "word", "langid" }` objects, in the same order as the whitespace-separated transcript; `language_counts` is a JSON object that summarizes the retained labels. Intermediate manifests also retain `source_audio`, `start_ms`, `end_ms`, `speakers`, and `utterance_ids` for traceability.

## Stage 1 — Indonesian source preparation

`01_prepare_indonesian_segments.py` uses separate preparation paths for the two corpora. Development MP3s are already clip-level, so they are converted directly to mono 16 kHz WAV. For each Jember recording, the script starts at every TSV row and emits every contiguous row extension whose combined transcript contains at least two words and whose TSV range lasts 3–40 seconds. It stops extending a start row when the next window would exceed 40 seconds, then advances the start by one row. The resulting windows intentionally overlap and preserve the original accented transcripts. Stage 1 does not perform VAD or forced alignment.

To build a balanced stage-1 review sample of up to 100 clips from each corpus:

```bash
uv run --project processing processing/01_prepare_indonesian_segments.py \
  --preview-per-dataset 100 \
  --output-dir processed_indonesia/window_preview/01_segments \
  --manifest processed_indonesia/window_preview/01_segments.csv
```

Then serve the repository and open the stage-1 viewer:

```bash
python3 -m http.server 8000
```

`http://127.0.0.1:8000/analysis_indonesia/segment_review.html?manifest=../processed_indonesia/window_preview/01_segments.csv`

The viewer can filter by corpus, play each generated WAV, and show its transcript, source range, and included TSV rows. Stage 1 never modifies the source TSV or source MP3s.

The supplied sources do not contain word-level language-ID labels, so each token is retained with the explicit `other` label rather than a fabricated Indonesian/Javanese split.

Common Voice Indonesian and Javanese clips are also prepared by default. Clips over 30 seconds are excluded. Stage 1 creates one deterministic synthetic Indonesian/Javanese code-switch example per eligible Javanese clip by appending one clip from each language with a uniformly random 0.5–1.0 second silent gap; joins over 30 seconds are skipped. Use `--commonvoice-code-switch-samples` to request fewer joins (or `0` to omit them) and `--commonvoice-seed` to reproduce a pairing.

Key options: `--jember-manifest`, `--jember-audio-dir`, `--development-manifest`, `--development-audio-dir`, `--output-dir`, and `--manifest`.

## Stage 2 — Silero voice activity detection

`02_vad_trim.py` reads `01_segments.csv`, decodes each source WAV to mono 16 kHz PCM through `ffmpeg`, and passes the samples to Silero VAD. A speech probability threshold of `0.5` is used by default. VAD is a quality gate only: if any speech is detected, the complete original timestamp segment is copied unchanged. This preserves pauses and keeps every transcript aligned with its audio.

Rows containing no detected speech are kept in `02_vad.csv` with `filtered_out=True`; rows with speech are written unchanged to `02_vad/` and marked `filtered_out=False`. Clips are also rejected when their transcript has fewer than 2 words or their source audio is shorter than 3 seconds. Every excluded row has a `filter_reason` of `no_vad_speech`, `too_few_words`, or `too_short`, so rejected material remains inspectable. The script decodes with `ffmpeg` rather than TorchCodec to avoid macOS audio-library compatibility issues.

Key options: `--threshold` (default `0.5`), `--min-words` (default `2`), `--min-seconds` (default `3`), `--manifest`, `--output-dir`, and `--output-manifest`.

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

### Qwen3-ASR all-linear LoRA training

`05_train_qwen3_asr.py` fine-tunes `Qwen/Qwen3-ASR-1.7B` with LoRA attached
to every linear module in both the audio encoder and text decoder. It reuses
the same augmentation-family-safe evaluation split and Jember training cap as
the Whisper run. Indonesian rows receive Qwen's `Indonesian` language prefix;
Javanese and code-switched rows use `None` because Qwen3-ASR does not advertise
Javanese as a supported language.

Install FlashAttention 2 separately on the CUDA training host, then run:

```bash
uv sync --project processing
uv run --project processing python processing/05_train_qwen3_asr.py \
  --config processing/runs/first_full_run_data_mix_commonvoice_jember_1200_qwen3_asr.yaml
```

The checkpoints under `best_lora/`, `best_wer/`, and `last_lora/` are PEFT
adapters and include processor files plus metadata naming the base model and
all targeted modules.

Re-run fresh greedy decoding from the best-WER adapter with:

```bash
uv run --project processing python processing/09_evaluate_qwen3_asr.py
```

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

### Evaluate a checkpoint on Common Voice Indonesian and Javanese

`09_evaluate_commonvoice.py` samples the same number of usable clips from the
local Common Voice Indonesian test split and Javanese spontaneous-speech
corpus, then runs the normal Whisper `transcribe` inference path. It writes a
single CSV with dataset name, audio path, original and predicted transcripts,
per-clip WER, and edit counts; it prints separate average-clip and corpus WER
summaries for Indonesian, Javanese, and their combined set.

```bash
uv run --project processing python processing/09_evaluate_commonvoice.py \
  --checkpoint processed_indonesia/window_preview_noise/05_whisper_small_full_eos4/best_wer \
  --subset-per-dataset 100 \
  --batch-size 8
```

The default output is `<checkpoint>/commonvoice_evaluation.csv`. Serve the
repository root, open `evaluation_full.html`, select that CSV, and set **Audio
base** to `.` so the CSV's repository-relative audio paths play correctly.
`--batch-size` defaults to 1; clips of 30 seconds or less are decoded together,
using greedy decoding. Longer clips retain Whisper's windowed single-clip
greedy inference path.

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

## Trained-model inference

`08_inference.py` loads a LoRA adapter from a training run and evaluates audio
listed in a CSV. The input CSV must be supplied explicitly with `--csv`; model,
checkpoint, device, and output settings can come from a named-run YAML file.
For example:

```bash
uv run --project processing python processing/08_inference.py \
  --config processing/runs/first_full_run_gpu.yaml \
  --csv experiments/indonesian_dev_long_clip_alignment/output/corrected_above30seconds.csv
```

The script uses Whisper's long-form `transcribe()` loop, so recordings longer
than 30 seconds are processed across successive windows rather than truncated.
It writes `results_{run_name}.csv` containing the reference, audio path,
prediction, and clip WER, then prints both average clip WER and corpus WER.
