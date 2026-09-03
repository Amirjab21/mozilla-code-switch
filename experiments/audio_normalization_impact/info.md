# Audio normalization impact

This experiment is about finding out whether audio normalization in
[`processing/03_normalize_audio.py`](../../processing/03_normalize_audio.py)
impacts the word error rate (WER).

The experiment will run on 300 examples, using only the development dataset.

It will first run
[`processing/02_vad_trim.py`](../../processing/02_vad_trim.py) on the clips to
produce the stage 02 CSV.

It will then create two copies of the audio:

1. A normalized copy produced using
   [`processing/03_normalize_audio.py`](../../processing/03_normalize_audio.py).
2. An unnormalized copy.

The wav2vec2 model will be run on both copies, and the results will be presented
in a side-by-side HTML comparison viewer.

The HTML viewer will include:

- Aggregate statistics for both groups.
- A list of examples showing both audio waveforms, generated using librosa.
- The WER for each version.
- The model transcript for each version.
- The ground-truth transcript.

All files should be saved inside this experiment subfolder where possible. If
the processing pipeline requires files to be stored under `processed_indonesia`,
they should be placed in a subfolder named `audio_normalization_impact`.

## Run

Prepare 300 development clips, normalize one copy, evaluate Wav2Vec2 on both,
and open the comparison viewer:

```bash
uv sync --project processing
uv run --project processing python experiments/audio_normalization_impact/run_experiment.py
```

Prepare audio only (skip Wav2Vec2):

```bash
uv run --project processing python experiments/audio_normalization_impact/run_experiment.py --skip-inference
```

Serve the repository root and open
[`comparison.html`](comparison.html?config=../../processed_indonesia/audio_normalization_impact/dataset.json).

The run stores its manifests, both audio sets, Wav2Vec2 results, and
librosa-generated waveform envelopes under
`processed_indonesia/audio_normalization_impact/`. Existing evaluation CSVs are
reused; pass `--force-inference` to regenerate them.
