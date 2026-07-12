#!/usr/bin/env python3
"""Tạo PNG cutout cột điện cho breakout ở bìa chương 1 (ui/assets/pylon-cutout.png).

Nguồn: ui/assets/quote.jpg — cột điện đen trên nền trời xanh dusk, tách bằng ngưỡng độ sáng.
Chạy từ gốc repo: python3 scripts/make_pylon_cutout.py [threshold]
Cần: opencv-python (cv2), numpy.
"""
import cv2, numpy as np, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "ui/assets/quote.jpg")
OUT = os.path.join(ROOT, "ui/assets/pylon-cutout.png")

img = cv2.imread(SRC)
assert img is not None, SRC

# Crop quanh cột chính + xà (giữ dây nối vào cột)
x0, x1, y0, y1 = 600, 1040, 86, 1000
crop = img[y0:y1, x0:x1].copy()
b = crop[..., 0].astype(int); g = crop[..., 1].astype(int); r = crop[..., 2].astype(int)
maxc = np.maximum(np.maximum(b, g), r)

# Cột = tối; trời xanh = sáng hơn. Ngưỡng theo maxchannel.
T = int(sys.argv[1]) if len(sys.argv) > 1 else 68
mask = (maxc < T).astype(np.uint8) * 255

k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)

# Giữ các cụm đủ lớn (bỏ đốm mây tối lẻ tẻ)
n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
keep = np.zeros_like(mask)
for i in range(1, n):
    if stats[i, cv2.CC_STAT_AREA] > 500:
        keep[lab == i] = 255
mask = keep

# Lấp CHỈ lỗ nhỏ bên trong silhouette (đốm phản sáng), chừa vùng trời lớn bị cáp bao
ff = mask.copy()
hh, ww = ff.shape
m2 = np.zeros((hh + 2, ww + 2), np.uint8)
cv2.floodFill(ff, m2, (0, 0), 255)
holes = cv2.bitwise_not(ff)
nh, hlab, hstats, _ = cv2.connectedComponentsWithStats(holes, 8)
small = np.zeros_like(mask)
for i in range(1, nh):
    if hstats[i, cv2.CC_STAT_AREA] < 700:
        small[hlab == i] = 255
mask = cv2.bitwise_or(mask, small)
mask = cv2.GaussianBlur(mask, (3, 3), 0)

# Trim về bbox của alpha
ys, xs = np.where(mask > 12)
mx0, mx1, my0, my1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
crop = crop[my0:my1, mx0:mx1]; mask = mask[my0:my1, mx0:mx1]

crop = cv2.convertScaleAbs(crop, alpha=1.12, beta=-6)
rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
rgba[..., 3] = mask
cv2.imwrite(OUT, rgba)
print(f"saved {OUT} size={rgba.shape[1]}x{rgba.shape[0]} T={T}")
