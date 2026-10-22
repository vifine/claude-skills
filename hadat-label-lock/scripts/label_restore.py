"""Restore label text/logos on a generated bottle from the Hero0-locked render, keeping the
generated bottle edges, gloss and shading.

usage: python3 label_restore.py gen.png locked.png locked_masks.npz out.png [--hmin 700] [--erode 17]
  gen.png            raw Weave generation (any size)
  locked.png         bottle_lock.py output for the same scene (bottle placed from Hero0)
  locked_masks.npz   masks written by bottle_lock.py next to locked.png (Mb, hgt)
  --hmin             hero row below which the body is replaced (below the collar / pump)
  --erode            px kept from the generation along the silhouette (odd)
"""
import argparse, numpy as np, cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('gen'); ap.add_argument('locked'); ap.add_argument('masks'); ap.add_argument('out')
ap.add_argument('--hmin', type=float, default=700); ap.add_argument('--erode', type=int, default=17)
a = ap.parse_args()

G8 = np.asarray(Image.open(a.gen).convert('RGB')); G = G8.astype(np.float32)
H, W = G8.shape[:2]
O8 = cv2.resize(np.asarray(Image.open(a.locked).convert('RGB')), (W, H), interpolation=cv2.INTER_CUBIC); O = O8.astype(np.float32)
m = np.load(a.masks)
Mb = cv2.resize(m['Mb'].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
hgt = cv2.resize(m['hgt'], (W, H))

body = ((Mb > 0) & (hgt > a.hmin)).astype(np.uint8)
reg = cv2.erode(body, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.erode, a.erode)))
# thin band at the edge: remove generated text strokes (median) but keep the edge itself
band = (cv2.erode(body, np.ones((5, 5), np.uint8)) > 0) & (reg == 0)
wb = cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.2)[..., None]
G2 = G * (1 - wb) + cv2.medianBlur(G8, 9).astype(np.float32) * wb
# label detail from the locked render, low-frequency shading from the generation
low = lambda x8: cv2.GaussianBlur(cv2.medianBlur(x8, 41).astype(np.float32), (0, 0), 12)
F = np.clip(O + (low(G8) - low(O8)), 0, 255)
w = cv2.GaussianBlur(reg.astype(np.float32), (0, 0), 2.5)[..., None]
out = G2 * (1 - w) + F * w
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out)
print('done', a.out)
