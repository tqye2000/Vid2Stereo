#!/usr/bin/env bash

set -euo pipefail

INPUT="${1:-examples/paradise_12.mp4}"
OUTPUT="${2:-examples/paradise_512.mp4}"

if command -v ffmpeg >/dev/null 2>&1; then
	FFMPEG_BIN="ffmpeg"
elif command -v ffmpeg.exe >/dev/null 2>&1; then
	FFMPEG_BIN="ffmpeg.exe"
elif [ -x "${HOME}/miniforge3/bin/ffmpeg" ]; then
	FFMPEG_BIN="${HOME}/miniforge3/bin/ffmpeg"
else
	echo "ERROR: ffmpeg not found on PATH."
	if grep -qi microsoft /proc/version 2>/dev/null; then
		echo "If using Miniforge, try: source ~/miniforge3/etc/profile.d/conda.sh && conda activate base"
		echo "Or install in WSL: sudo apt update && sudo apt install -y ffmpeg"
	else
		echo "Install ffmpeg and ensure it is on PATH."
	fi
	exit 127
fi

"$FFMPEG_BIN" -i "$INPUT" -vf scale=512:512 -c:a copy "$OUTPUT"

#Center-crop then scale (fills frame):
#ffmpeg -i "$INPUT" -vf "crop=min(iw,ih):min(iw,ih),scale=512:512" -c:a copy "$OUTPUT"

# Keep full frame and pad to square:
#ffmpeg -i "$INPUT" -vf "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2" -c:a copy "$OUTPUT"
