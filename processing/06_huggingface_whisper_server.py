#!/usr/bin/env python3
"""Serve the repository and local Hugging Face Whisper transcription endpoint."""

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
from transformers import WhisperForConditionalGeneration, WhisperProcessor


MODEL_IDS = {
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
}


class WhisperService:
    def __init__(self, device: str) -> None:
        self.device = torch.device(device)
        self.loaded: dict[str, tuple[WhisperProcessor, WhisperForConditionalGeneration]] = {}
        self.lock = threading.Lock()

    def transcribe(self, audio_path: Path, variant: str) -> str:
        if variant not in MODEL_IDS:
            raise ValueError(f"Unsupported Whisper variant: {variant}")
        with self.lock:
            if variant not in self.loaded:
                print(f"Loading Hugging Face {MODEL_IDS[variant]} on {self.device}…")
                processor = WhisperProcessor.from_pretrained(MODEL_IDS[variant])
                model = WhisperForConditionalGeneration.from_pretrained(MODEL_IDS[variant]).to(self.device).eval()
                self.loaded[variant] = (processor, model)
            processor, model = self.loaded[variant]
            audio = whisper.load_audio(str(audio_path))
            features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
            with torch.no_grad():
                token_ids = model.generate(features, task="transcribe")
            return processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()


def select_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_handler(root: Path, service: WhisperService):
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
                self.send_json(HTTPStatus.OK, {"transcript": transcript, "model": request.get("model", "small")})
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
    args = parser.parse_args()
    root = args.root.resolve()
    service = WhisperService(select_device(args.device))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, service))
    print(f"Serving {root} at http://{args.host}:{args.port} (Hugging Face Whisper device: {service.device})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
