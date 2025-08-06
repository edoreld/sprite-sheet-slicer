#!/usr/bin/env python3
"""
Slice circular sprites from a white-background sprite sheet.

Requirements:
  - Pillow (PIL): pip install pillow
  - NumPy:        pip install numpy

Usage:
  python slice_circular_sprites.py input.png --out sprites --pad 6 --white-thresh 245 --min-area 200 --feather 1.5
"""

import argparse
import os
from collections import deque

import numpy as np
from PIL import Image, ImageFilter


def parse_args():
    p = argparse.ArgumentParser(description="Slice circular sprites from a white-background sheet.")
    p.add_argument("image", help="Path to the sprite sheet (white background).")
    p.add_argument("--out", default="slices", help="Output directory for extracted sprites (default: slices).")
    p.add_argument("--white-thresh", type=int, default=245,
                   help="0–255. Pixels >= this in ALL channels are treated as white background. (default: 245)")
    p.add_argument("--min-area", type=int, default=100,
                   help="Minimum blob area in pixels to keep (default: 100).")
    p.add_argument("--max-area", type=int, default=0,
                   help="Maximum blob area in pixels to keep. 0 disables upper filter (default: 0).")
    p.add_argument("--pad", type=int, default=4,
                   help="Padding (pixels) around each crop (default: 4).")
    p.add_argument("--feather", type=float, default=1.0,
                   help="Gaussian blur radius for alpha edge feathering (default: 1.0, set 0 to disable).")
    p.add_argument("--sort", choices=["x", "y", "area"], default="x",
                   help="Sort slices by left-edge x, top-edge y, or area (default: x).")
    return p.parse_args()


def load_image(path):
    img = Image.open(path).convert("RGBA")  # keep alpha if present
    return img


def make_foreground_mask(img, white_thresh=245):
    """
    Return boolean mask where True = non-white (foreground).
    A pixel is 'white' if all channels >= white_thresh.
    """
    arr = np.asarray(img)  # HxWx4
    rgb = arr[..., :3]
    # white if all channels >= threshold
    white = (rgb >= white_thresh).all(axis=-1)
    fg = ~white
    # Also strip fully transparent if input has alpha
    if arr.shape[-1] == 4:
        alpha = arr[..., 3] > 0
        fg = fg & alpha
    return fg


def connected_components(mask):
    """
    4-connected component labeling using BFS.
    Returns: labels (int array), num_labels, bbox list [(miny,minx,maxy,maxx), ...], areas list
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label_id = 0
    bboxes = []
    areas = []

    visited = np.zeros_like(mask, dtype=bool)
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]

    for y in range(h):
        for x in range(w):
            if mask[y, x] and not visited[y, x]:
                label_id += 1
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                labels[y, x] = label_id

                miny = maxy = y
                minx = maxx = x
                area = 0

                while q:
                    cy, cx = q.popleft()
                    area += 1
                    if cy < miny: miny = cy
                    if cy > maxy: maxy = cy
                    if cx < minx: minx = cx
                    if cx > maxx: maxx = cx
                    for dy, dx in dirs:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            labels[ny, nx] = label_id
                            q.append((ny, nx))

                bboxes.append((miny, minx, maxy, maxx))
                areas.append(area)

    return labels, label_id, bboxes, areas


def crop_and_save_components(img, labels, bboxes, areas, out_dir, pad=4, min_area=100, max_area=0, feather=1.0, sort="x"):
    os.makedirs(out_dir, exist_ok=True)
    arr = np.asarray(img)  # HxWx4

    items = []
    for idx, (bbox, area) in enumerate(zip(bboxes, areas), start=1):
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        miny, minx, maxy, maxx = bbox
        items.append((idx, bbox, area, (minx, miny)))  # store for sorting

    if sort == "x":
        items.sort(key=lambda t: t[3][0])  # by left x
    elif sort == "y":
        items.sort(key=lambda t: t[3][1])  # by top y
    elif sort == "area":
        items.sort(key=lambda t: t[2], reverse=True)

    saved = 0
    h, w = arr.shape[:2]

    for out_idx, (idx, (miny, minx, maxy, maxx), area, _) in enumerate(items, start=1):
        # expand with padding
        y0 = max(miny - pad, 0)
        x0 = max(minx - pad, 0)
        y1 = min(maxy + pad + 1, h)
        x1 = min(maxx + pad + 1, w)

        crop = arr[y0:y1, x0:x1].copy()

        # alpha: based on pixels that belong to this component label
        comp_mask = (labels[y0:y1, x0:x1] == idx).astype(np.uint8) * 255
        alpha = Image.fromarray(comp_mask, mode="L")

        # Optional feathering for smoother edges
        if feather > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(radius=float(feather)))

        # Apply alpha to crop (regardless of original alpha)
        crop_pil = Image.fromarray(crop, mode="RGBA")
        r, g, b, _ = crop_pil.split()
        crop_pil = Image.merge("RGBA", (r, g, b, alpha))

        out_path = os.path.join(out_dir, f"sprite_{out_idx:03d}.png")
        crop_pil.save(out_path)
        saved += 1

    return saved


def main():
    args = parse_args()
    img = load_image(args.image)
    fg_mask = make_foreground_mask(img, white_thresh=args.white_thresh)
    labels, n, bboxes, areas = connected_components(fg_mask)
    saved = crop_and_save_components(
        img, labels, bboxes, areas,
        out_dir=args.out,
        pad=args.pad,
        min_area=args.min_area,
        max_area=args.max_area,
        feather=args.feather,
        sort=args.sort,
    )
    print(f"Found {n} blobs; saved {saved} sprites to '{args.out}'.")


if __name__ == "__main__":
    main()
