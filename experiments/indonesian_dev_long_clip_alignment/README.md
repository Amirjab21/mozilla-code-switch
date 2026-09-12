# Indonesian development long-clip alignment experiment

This experiment selects Indonesian-development audio longer than 30 seconds,
splits each source at the midpoint of its longest eligible silence, and uses
`indonesian-nlp/wav2vec2-indonesian-javanese-sundanese` to infer the transcript
boundary. The first half contributes its final predicted word and the second
half contributes its initial predicted word. Every reference word boundary is
scored using the average normalized Levenshtein distance of those two matches.

Run a small preview:

```bash
uv run --project processing \
  python experiments/indonesian_dev_long_clip_alignment/run_experiment.py \
  --limit 5 \
  --device auto
```

Remove `--limit 5` to process all long development clips. Generated WAVs, CSV,
JSON diagnostics, and viewer data are written under this experiment's `output/`
directory.

To review and edit the aligned transcripts, run the writable review server from
the repository root:

```bash
python3 experiments/indonesian_dev_long_clip_alignment/serve_review.py
```

Open the viewer URL printed by the server (through the SageMaker port proxy when
running in Code Editor). The **Save corrected CSV** button writes one row per
split clip to `output/corrected_above30seconds.csv`, with the columns
`original_audio_path`, `corrected_transcript`, and `audio_path`. The separate
download button works when the viewer is hosted by a read-only static server.
