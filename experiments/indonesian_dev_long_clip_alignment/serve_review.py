#!/usr/bin/env python3
"""Serve the alignment viewer and persist transcript corrections."""

from __future__ import annotations

import argparse
import csv
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
OUTPUT_PATH = EXPERIMENT_DIR / "output" / "corrected_above30seconds.csv"
FIELDS = ("original_audio_path", "corrected_transcript", "audio_path")


class ReviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPOSITORY_ROOT), **kwargs)

    def do_POST(self) -> None:  # noqa: N802 - HTTP handler API
        if not urlsplit(self.path).path.rstrip("/").endswith("/save-corrections"):
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            rows = payload["rows"]
            if not isinstance(rows, list):
                raise ValueError("rows must be a list")

            clean_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("each row must be an object")
                clean_rows.append({field: str(row.get(field, "")) for field in FIELDS})

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")
            with temporary_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(clean_rows)
            temporary_path.replace(OUTPUT_PATH)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_error(400, str(error))
            return

        result = json.dumps(
            {
                "path": str(OUTPUT_PATH.relative_to(REPOSITORY_ROOT)),
                "row_count": len(clean_rows),
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(result)))
        self.end_headers()
        self.wfile.write(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    viewer = (
        f"http://127.0.0.1:{args.port}/experiments/"
        "indonesian_dev_long_clip_alignment/review.html"
    )
    print(f"Serving repository from {REPOSITORY_ROOT}")
    print(f"View and edit corrections: {viewer}")
    print(f"Corrections will be saved to {OUTPUT_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
