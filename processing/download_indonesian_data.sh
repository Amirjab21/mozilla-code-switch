#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
S3_SOURCE="${1:-s3://ait-ai-storage-prod/indonesian-data/}"
LOCAL_DESTINATION="$REPOSITORY_DIR/indonesian_data"

mkdir -p "$LOCAL_DESTINATION"

echo "Downloading $S3_SOURCE to $LOCAL_DESTINATION"
aws s3 sync "$S3_SOURCE" "$LOCAL_DESTINATION" --only-show-errors

echo "Download complete. Local files:"
find "$LOCAL_DESTINATION" -type f | wc -l
