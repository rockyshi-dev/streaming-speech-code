#!/bin/bash
# Fetch sherpa-onnx's streaming WebSocket server (python-api-examples, tag v2.0.1, Apache-2.0) for
# ch15/sherpa_server.py, and check it against the hashes the measurements were made with.
#
#   code/serve/ch15/fetch_vendor.sh            # into $BOOK_DATA/vendor/sherpa-onnx-v2.0.1, where sherpa_server.py reads them
set -euo pipefail
source "$(dirname "$0")/../../tools/roots.sh"
DEST=$BOOK_DATA/vendor/sherpa-onnx-v2.0.1
BASE=https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v2.0.1/python-api-examples
mkdir -p "$DEST"
while read -r sha name; do
  curl -fsSL "$BASE/$name" -o "$DEST/$name.part"
  echo "$sha  $DEST/$name.part" | sha256sum -c --quiet
  mv "$DEST/$name.part" "$DEST/$name"
done <<'LIST'
38f989bfa2200b3d68398cb80043ff88bac84fceedf46e0900678669ed90b42a streaming_server.py
a78a7df73d44b7c8a6793a6454c10b948d245460d9f1ec69612eb2785e6901a0 http_server.py
LIST
echo "fetched into $DEST"
