#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Convert every .wav file under the given directories to 32 kbps mono Opus.
#
# Usage: ./wav_to_opus.sh DIR [DIR...]
#
# NAME.wav becomes NAME.opus in the same place. The original is then moved to
# a _wav_backups directory next to DIR, keeping its path below DIR:
#
#   inputs/USA_Day01/a.wav -> inputs/USA_Day01/a.opus
#                             inputs/_wav_backups/USA_Day01/a.wav
#
# A WAV is moved only after its Opus file is written and the durations match.
# Files whose .opus or backup already exists are skipped.
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 DIR [DIR...]" >&2
    exit 2
fi

for dir in "$@"; do
    if [[ ! -d "$dir" ]]; then
        echo "Not a directory: $dir" >&2
        exit 2
    fi
done

converted=0
skipped=0

duration() {
    ffprobe -v error -show_entries format=duration -of csv=p=0 "$1"
}

for dir in "$@"; do
    dir="${dir%/}"
    resolved="$(realpath "$dir")"
    backup_dir="$(dirname "$resolved")/_wav_backups/$(basename "$resolved")"

    while IFS= read -r -d '' wav; do
        opus="${wav%.*}.opus"
        tmp="${wav%.*}.tmp.opus"
        backup="$backup_dir/${wav#"$dir"/}"

        if [[ -e "$opus" || -e "$backup" ]]; then
            echo "Skipping, target already exists: $wav" >&2
            skipped=$((skipped + 1))
            continue
        fi

        ffmpeg -nostdin -hide_banner -loglevel error -y -i "$wav" \
            -vn -c:a libopus -b:a 32k -ac 1 "$tmp"

        if ! awk -v a="$(duration "$wav")" -v b="$(duration "$tmp")" \
            'BEGIN { d = a - b; if (d < 0) d = -d; exit !(d < 0.5) }'; then
            echo "Duration mismatch, keeping WAV: $wav" >&2
            rm -f "$tmp"
            exit 1
        fi

        mv "$tmp" "$opus"
        mkdir -p "$(dirname "$backup")"
        mv -n "$wav" "$backup"
        converted=$((converted + 1))
        echo "$opus ($(du -h "$backup" | cut -f1) -> $(du -h "$opus" | cut -f1))"
    done < <(find "$dir" -type f -iname '*.wav' -print0 | sort -z)
done

echo "Converted $converted file(s), skipped $skipped."
[[ $skipped -eq 0 ]]
