# Whisper token-language fork

This local fork retains Whisper's normal transcription head and adds a second classifier head to the final text-decoder state. For every emitted Whisper BPE token it produces one of `eng`, `spa`, `mixed`, or `other`, plus a confidence. It does not add language labels to Whisper's text vocabulary or force language labels to influence transcription.

## Contents

- `model.py` defines `WhisperTokenLID`, loads official Whisper weights with a freshly initialised `language_head`, and saves/loads fork checkpoints.
- `model_lora.py` applies PEFT LoRA to every linear projection in Whisper's encoder and decoder. The language-ID head is excluded from LoRA but remains normally trainable and is saved with the adapter.
- `labels.py` converts the existing `word_langids` JSON field into one target per Whisper BPE token. `eng&spa` is ignored for the auxiliary loss because it is an undetermined corpus label.
- `decode.py` supplies greedy inference and returns BPE-level and pooled word-level language predictions.

## First use

Run from the repository root. The first command downloads the official Small checkpoint if it is not already under `models/whisper/`.

```bash
uv run --project processing python - <<'PY'
import whisper
from models.whisper_lid import load_token_lid_model

model = load_token_lid_model("small", download_root="models/whisper")
print(model.language_labels)
PY
```

The new head is random until fine-tuned; its predictions are not meaningful before training.

## LoRA variant

Synchronise the processing environment once to install PEFT, then load the LoRA version in place of the full-finetuning model:

```bash
uv sync --project processing
uv run --project processing python - <<'PY'
from models.whisper_lid import load_token_lid_lora_model

model = load_token_lid_lora_model(
    "small",
    rank=16,
    alpha=32,
    dropout=0.05,
    download_root="models/whisper",
)
model.print_trainable_parameters()
PY
```

This freezes the original Whisper weights. The trainable parameters are the LoRA matrices in the encoder/decoder projections plus the full `decoder.language_head`; the embedding and normal Whisper token-output weights remain frozen.

## Training contract

Teacher-force Whisper as usual. For each reference BPE token, add its language target returned by `make_token_language_targets()`. The language target must be shifted exactly as the ASR next-token target is shifted: the decoder state that predicts transcript token *t+1* also predicts the language of token *t+1*.

```python
asr_loss = cross_entropy(token_logits, next_token_ids, ignore_index=-100)
lid_loss = cross_entropy(language_logits, next_language_ids, ignore_index=-100)
loss = asr_loss + 0.3 * lid_loss
```

Start with greedy decoding. The fork deliberately does not expose beam search yet, because beam candidates must carry and reorder parallel language-label histories.
