#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${UI_CAPTURE_URL:-http://127.0.0.1:8899}"
OUT_DIR="${UI_CAPTURE_OUT:-docs/scrollytelling_audit/_ui_current}"
VIEWPORT="${UI_CAPTURE_VIEWPORT:-1440,900}"
WAIT_MS="${UI_CAPTURE_WAIT_MS:-900}"

mkdir -p "$OUT_DIR"

if ! node -e "require('playwright')" >/dev/null 2>&1; then
  PW_BIN="$(npm exec --yes --package=playwright -- which playwright)"
  export NODE_PATH="$(cd "$(dirname "$PW_BIN")/.." && pwd)"
fi

UI_CAPTURE_URL="$BASE_URL" \
UI_CAPTURE_OUT="$OUT_DIR" \
UI_CAPTURE_VIEWPORT="$VIEWPORT" \
UI_CAPTURE_WAIT_MS="$WAIT_MS" \
node scripts/capture_ui_current.cjs

python3 - "$OUT_DIR" <<'PY'
import math
import pathlib
import sys
from PIL import Image, ImageDraw, ImageFont

out_dir = pathlib.Path(sys.argv[1])
names = [
    "hero.png",
    "intro.png",
    "cover.png",
    "article.png",
    "stages.png",
    "bridge.png",
    "state.png",
    "network.png",
    "quote.png",
    "electric.png",
    "water.png",
    "stats.png",
    "floor.png",
    "debate.png",
    "final.png",
]

thumb_w = 360
thumb_h = 225
label_h = 34
gap = 18
cols = 4
rows = math.ceil(len(names) / cols)
sheet_w = cols * thumb_w + (cols + 1) * gap
sheet_h = rows * (thumb_h + label_h) + (rows + 1) * gap

sheet = Image.new("RGB", (sheet_w, sheet_h), "#0b161b")
draw = ImageDraw.Draw(sheet)
try:
    font = ImageFont.truetype("Arial.ttf", 16)
except Exception:
    font = ImageFont.load_default()

for i, name in enumerate(names):
    path = out_dir / name
    if not path.exists():
        continue
    img = Image.open(path).convert("RGB")
    img.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
    x = gap + (i % cols) * (thumb_w + gap)
    y = gap + (i // cols) * (thumb_h + label_h + gap)
    frame = Image.new("RGB", (thumb_w, thumb_h), "#12242b")
    frame.paste(img, ((thumb_w - img.width) // 2, (thumb_h - img.height) // 2))
    sheet.paste(frame, (x, y))
    draw.text((x, y + thumb_h + 10), name, fill="#eaf3f5", font=font)

sheet.save(out_dir / "_contact.jpg", quality=92)
print(f"contact -> {out_dir / '_contact.jpg'}")
PY
