#!/usr/bin/env python3
"""Serve the repository and local OpenAI Whisper transcription endpoint."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import whisper
WHISPER_VARIANTS = {"small", "medium", "large"}
WAV2VEC2_VARIANT = "wav2vec2-indonesian-javanese-sundanese"
WAV2VEC2_MODEL_ID = "indonesian-nlp/wav2vec2-indonesian-javanese-sundanese"
# Do not use Whisper's stochastic fallback-temperature schedule for the
# on-demand comparison control; it must be repeatable for the same clip.
TRANSCRIBE_OPTIONS = {"task": "transcribe", "temperature": 0, "beam_size": 5}


class TranscriptionService:
    def __init__(self, device: str, model_dir: Path) -> None:
        self.device = torch.device(device)
        self.model_dir = model_dir
        self.loaded: dict[str, object] = {}
        self.lock = threading.Lock()

    def transcribe(self, audio_path: Path, variant: str) -> str:
        if variant in WHISPER_VARIANTS:
            return self.transcribe_whisper(audio_path, variant)
        if variant == WAV2VEC2_VARIANT:
            return self.transcribe_wav2vec2(audio_path)
        raise ValueError(f"Unsupported transcription model: {variant}")

    def transcribe_whisper(self, audio_path: Path, variant: str) -> str:
        with self.lock:
            if variant not in self.loaded:
                print(f"Loading OpenAI Whisper {variant} from {self.model_dir} on {self.device}…")
                self.loaded[variant] = whisper.load_model(variant, device=str(self.device), download_root=str(self.model_dir))
            model = self.loaded[variant]
            result = model.transcribe(str(audio_path), fp16=self.device.type == "cuda", verbose=False, **TRANSCRIBE_OPTIONS)
            return result["text"].strip()

    def transcribe_wav2vec2(self, audio_path: Path) -> str:
        with self.lock:
            if WAV2VEC2_VARIANT not in self.loaded:
                print(f"Loading {WAV2VEC2_MODEL_ID} on {self.device}…")
                from transformers import AutoModelForCTC, Wav2Vec2Processor

                self.loaded[WAV2VEC2_VARIANT] = (
                    Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_ID),
                    AutoModelForCTC.from_pretrained(WAV2VEC2_MODEL_ID).to(self.device).eval(),
                )
            processor, model = self.loaded[WAV2VEC2_VARIANT]
            audio = whisper.load_audio(str(audio_path))
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(
                    inputs.input_values.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None,
                ).logits
            return processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()

    def metadata(self, variant: str) -> dict[str, str]:
        if variant == WAV2VEC2_VARIANT:
            return {"engine": "transformers-wav2vec2", "device": str(self.device), "model": WAV2VEC2_MODEL_ID}
        return {
            "engine": "openai-whisper",
            "device": str(self.device),
            "model": variant,
            "model_dir": str(self.model_dir),
        }


def select_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_handler(root: Path, service: TranscriptionService):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def send_head(self):
            """Serve static files with byte ranges, allowing long MP3s to seek."""
            path = self.translate_path(self.path)
            if os.path.isdir(path):
                return super().send_head()
            try:
                source = open(path, "rb")
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return None

            self.byte_range: tuple[int, int] | None = None
            size = os.fstat(source.fileno()).st_size
            range_header = self.headers.get("Range")
            if not range_header:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return source

            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
            if not match:
                source.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None
            start_text, end_text = match.groups()
            if not start_text and not end_text:
                source.close()
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                return None
            if start_text:
                start = int(start_text)
                end = int(end_text) if end_text else size - 1
            else:
                length = int(end_text)
                start = max(size - length, 0)
                end = size - 1
            if start >= size or end < start:
                source.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            end = min(end, size - 1)
            self.byte_range = (start, end)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            return source

        def copyfile(self, source, outputfile) -> None:
            if self.byte_range is None:
                super().copyfile(source, outputfile)
                return
            start, end = self.byte_range
            source.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    outputfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    # Browsers routinely cancel an in-progress range while
                    # seeking again; that is an expected client disconnect.
                    break
                remaining -= len(chunk)

        def send_json(self, status: int, payload: dict) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:
            if self.path != "/api/transcribe":
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                relative_path = Path(request["audio_path"])
                audio_path = (root / relative_path).resolve() if not relative_path.is_absolute() else relative_path.resolve()
                if not audio_path.is_relative_to(root) or not audio_path.is_file():
                    raise ValueError("audio_path must be an existing file below the repository root")
                transcript = service.transcribe(audio_path, request.get("model", "small"))
                self.send_json(HTTPStatus.OK, {"transcript": transcript, **service.metadata(request.get("model", "small"))})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except Exception as error:  # keep server errors visible in the page while retaining the terminal traceback
                print(f"Transcription failed: {error}")
                self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root to serve.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--whisper-model-dir", type=Path, default=Path("models/whisper"), help="Directory containing OpenAI Whisper .pt checkpoints.")
    args = parser.parse_args()
    root = args.root.resolve()
    model_dir = args.whisper_model_dir.resolve()
    service = TranscriptionService(select_device(args.device), model_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, service))
    print(f"Serving {root} at http://{args.host}:{args.port} (OpenAI Whisper device: {service.device}; models: {model_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
