# Learnings

- On clips with mixed audio, Whisper Medium tends to commit to one language and transcribes the other language almost completely incorrectly.

## Preview evaluation — Whisper Medium

- Evaluated clips: 49
- Aggregate WER: 43.2%

| Language ID | Reference words | WER |
| --- | ---: | ---: |
| `eng` | 1,679 | 31.0% |
| `spa` | 991 | 58.5% |
| `eng&spa` | 107 | 92.5% |
