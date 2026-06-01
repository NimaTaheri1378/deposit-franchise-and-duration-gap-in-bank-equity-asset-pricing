from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from deposit_duration.utils.manifest import write_manifest


def _image_stats(path: Path) -> dict[str, float | int | str | bool]:
    img = mpimg.imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        rgb = img[:, :, :3]
    elif img.ndim == 3:
        rgb = img
    else:
        rgb = np.repeat(img[:, :, None], 3, axis=2)
    gray = rgb.mean(axis=2)
    edge = 8
    border = np.concatenate(
        [
            gray[:edge, :].ravel(),
            gray[-edge:, :].ravel(),
            gray[:, :edge].ravel(),
            gray[:, -edge:].ravel(),
        ]
    )
    dark = gray < 0.18
    edge_dark_share = float((border < 0.18).mean())
    return {
        "file": str(path),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "std": float(np.nanstd(gray)),
        "dark_pixel_share": float(dark.mean()),
        "edge_dark_share": edge_dark_share,
        "blank_like": bool(np.nanstd(gray) < 0.01),
        "small_canvas": bool(img.shape[0] < 600 or img.shape[1] < 800),
        "possible_clipped_text": bool(edge_dark_share > 0.015),
    }


def audit_static_figures(figures_dir: str | Path, out_dir: str | Path | None = None) -> pd.DataFrame:
    figures_dir = Path(figures_dir)
    if out_dir is None:
        out_dir = figures_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sorted(figures_dir.rglob("*.png"))
    if not pngs:
        raise ValueError(f"No PNG figures found under {figures_dir}")
    audit = pd.DataFrame([_image_stats(path) for path in pngs])
    audit["pass"] = ~(audit["blank_like"] | audit["small_canvas"] | audit["possible_clipped_text"])
    audit.to_csv(out_dir / "visual_audit.csv", index=False)
    write_manifest(
        out_dir / "visual_audit_manifest.json",
        {
            "kind": "visual_audit",
            "figures_dir": str(figures_dir),
            "png_count": len(audit),
            "passing": int(audit["pass"].sum()),
            "failing": int((~audit["pass"]).sum()),
            "heuristics": {
                "blank_like": "image grayscale standard deviation below 0.01",
                "small_canvas": "height < 600 or width < 800",
                "possible_clipped_text": "dark pixel share in outer 8-pixel border above 1.5%",
            },
        },
    )
    return audit


def build_contact_sheet(
    figures_dir: str | Path,
    out_path: str | Path,
    thumb_width: int = 720,
    padding: int = 28,
    label_height: int = 36,
) -> Path:
    figures_dir = Path(figures_dir)
    pngs = sorted(figures_dir.rglob("*.png"))
    if not pngs:
        raise ValueError(f"No PNG figures found under {figures_dir}")
    thumbs = []
    for path in pngs:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumb = image.resize((thumb_width, max(1, int(image.height * ratio))))
        thumbs.append((path, thumb))
    cols = 2 if len(thumbs) > 1 else 1
    rows = int(np.ceil(len(thumbs) / cols))
    cell_h = max(thumb.height for _, thumb in thumbs) + label_height + padding
    width = cols * thumb_width + (cols + 1) * padding
    height = rows * cell_h + padding
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, thumb) in enumerate(thumbs):
        row, col = divmod(i, cols)
        x = padding + col * (thumb_width + padding)
        y = padding + row * cell_h
        draw.text((x, y), path.stem.replace("_", " "), fill=(30, 30, 30))
        sheet.paste(thumb, (x, y + label_height))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
