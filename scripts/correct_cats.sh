#!/usr/bin/env bash
# copy_by_list.sh
# Copy files whose names are listed in a text file from a source folder to a new folder.

set -Eeuo pipefail

usage() {
  echo "Usage: $0 <SOURCE_DIR> <DEST_DIR> <LIST_FILE>"
  echo "Example: $0 ./data/afhq_cat/test/cat ./data/afhq_cat/val/cat ./data/splits/filenames.txt"
  exit 1
}

[[ $# -eq 3 ]] || usage

SRC_DIR=$1
DEST_DIR=$2
LIST_FILE=$3

# Basic checks
[[ -d "$SRC_DIR" ]] || { echo "Source dir not found: $SRC_DIR" >&2; exit 2; }
[[ -f "$LIST_FILE" ]] || { echo "List file not found: $LIST_FILE" >&2; exit 3; }

mkdir -p "$DEST_DIR"

copied=0
skipped=0
missing=0

# Read list line-by-line (handles spaces; ignores blank lines and comments)
while IFS= read -r line || [[ -n "$line" ]]; do
  # strip potential CR from Windows line endings
  line=${line%$'\r'}
  [[ -z "$line" || "$line" == \#* ]] && continue

  src_path="$SRC_DIR/$line"
  base_name="$(basename "$line")"
  dest_path="$DEST_DIR/$base_name"

  if [[ -e "$src_path" ]]; then
    if [[ -e "$dest_path" ]]; then
      echo "Skip (exists): $base_name"
      ((skipped++))
    else
      # preserve timestamps/permissions; avoid following symlinks
      cp -p -- "$src_path" "$dest_path"
      echo "Copied: $base_name"
      ((copied++))
    fi
  else
    echo "Not found in source: $src_path" >&2
    ((missing++))
  fi
done < "$LIST_FILE"

echo "Done. Copied: $copied, Skipped: $skipped, Missing: $missing"
