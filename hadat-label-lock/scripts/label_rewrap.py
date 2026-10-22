"""Re-wrap a label onto a generated bottle at any tilt, keeping the generated bottle
(shape, light, gloss) and replacing only the label.

Pipeline (BEN1 112506, approved by Victoria 18.10.2026):
  1. rotate the scene so the bottle axis is vertical (pivot = base centre);
  2. erase the old label: rows y0..y1 between body edges are rebuilt from the clean
     strips just above / below the label (per-column vertical interpolation, feathered);
  3. wrap the flat label artwork (Label0 export) on a cylinder: theta = asin((x-cx)/R),
     u = uc + theta*R/k, v = (y - top)/k;
  4. overlay: ink (text, logos) multiplies the clean surface; grey fills -> silver satin;
     green -> gold foil (ramp driven by the old gold's column brightness in the source,
     plus soft diagonal sheen); magenta -> clear UV varnish (paper x0.955 + glint where the
     cylinder is brightest); thin paper-edge bevel at the label top / bottom;
  5. rotate back and composite only the erased band.

usage:
  python3 label_rewrap.py scene.png label_art.png out.png --angle 18.3 --pivot 1000,1640 \
      --edges 732.5,1232 --erase 922,1600 --k 0.745 --uc 532.7 --yc 1263 \
      [--patch 609,280,1338,240,15,17]

  scene.png      raw generation (native size, e.g. 2048x2048)
  label_art.png  flat label artwork export (e.g. 112506 Label0 1063:6965, 2084x806):
                 green = gold foil, magenta = UV varnish, grey fills = silver satin, rest = ink/paper
  --angle        deg; scene is rotated counter-clockwise by this to make the bottle upright
                 (bottle top leaning right -> positive). Measure from the clean body edge.
  --pivot        rotation centre (bottle base centre)
  --edges        body left,right x in the upright image (label band rows)
  --erase        rows (upright) to rebuild: a little above the old label top, below its bottom
  --k            scene px per artwork px (label height in scene / artwork height)
  --uc           artwork x that faces the camera (theta = 0)
  --yc           label centre row in the upright image
  --patch        dst_x,dst_y,src_x,src_y,w,h : copy a glyph inside the artwork (darken),
                 e.g. a letter missing in the export; repeatable
Writes out.png, out_upright.png (check view), out_compare.png (before / after crop).
"""
import argparse, numpy as np, cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('scene'); ap.add_argument('art'); ap.add_argument('out')
ap.add_argument('--angle', type=float, required=True)
ap.add_argument('--pivot', required=True)
ap.add_argument('--edges', required=True)
ap.add_argument('--erase', required=True)
ap.add_argument('--k', type=float, required=True)
ap.add_argument('--uc', type=float, required=True)
ap.add_argument('--yc', type=float, required=True)
ap.add_argument('--patch', action='append', default=[])
ap.add_argument('--bulge', type=float, default=0.0, help='label height shrink toward the bottle sides: h(theta)=h0*(1-bulge*(1-cos theta))')
ap.add_argument('--light_ink', action='store_true', help='dark label: grey artwork background = paper, light artwork pixels = light ink (no satin fill from grey)')
ap.add_argument('--top0', type=float, default=None, help='old label top edge row at the camera-facing centre')
ap.add_argument('--bot0', type=float, default=None)
ap.add_argument('--arc_top', type=float, default=0.0, help='px the top edge rises toward the bottle sides (perspective arc)')
ap.add_argument('--arc_bot', type=float, default=0.0)
ap.add_argument('--corner_slope', type=float, default=None, help='max slope of the edge arcs near the bottle sides (e.g. 0.8): keeps sharp label corners instead of the arc curling up into the silhouette')
ap.add_argument('--edge_light', type=float, default=6.0, help='brightness of the thin paper-edge highlight along the top/bottom label edge (0 = no light line)')
ap.add_argument('--paper', choices=['body', 'scene'], default='body',
                help='body: label paper = erased bottle surface (white bottles); scene: paper tone taken from the old label itself (dark bottles with a light label)')
a = ap.parse_args()

CEN = tuple(map(float, a.pivot.split(',')))
Y_H = Image.open(a.art).size[1] * a.k   # label height in scene px
XL, XR = map(float, a.edges.split(','))
Y0, Y1 = map(int, a.erase.split(','))
S_img = Image.open(a.scene).convert('RGB')
RO = np.asarray(S_img.rotate(a.angle, center=CEN, resample=Image.BICUBIC)).astype(np.float32)
H, W = RO.shape[:2]
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------- 1. erase old label
T0, T1 = (Y0 - 22, Y0 - 1), (Y1 + 1, Y1 + 13)
top_s = cv2.GaussianBlur(np.median(RO[T0[0]:T0[1]], axis=0)[None], (0, 0), 2)[0]
bot_s = cv2.GaussianBlur(np.median(RO[T1[0]:T1[1]], axis=0)[None], (0, 0), 2)[0]
C = RO.copy()
for y in range(Y0, Y1):
    t = np.clip((y - T0[1]) / (T1[0] - T0[1]), 0, 1)
    C[y] = top_s * (1 - t) + bot_s * t
C += np.random.default_rng(1).normal(0, 0.8, C.shape)
M = np.zeros((H, W), np.float32); M[Y0:Y1, int(XL) + 2:int(XR) - 1] = 1
M = cv2.GaussianBlur(M, (0, 0), 1.2)
ry = np.zeros(H, np.float32); f0 = min(28, (Y1 - Y0) // 4)
ry[Y0:Y1] = 1; ry[Y0:Y0 + f0] = np.linspace(0, 1, f0); ry[Y1 - f0 // 1:Y1] = np.linspace(1, 0, f0)
M *= ry[:, None]
C = RO * (1 - M[..., None]) + C * M[..., None]
if a.paper == 'scene':
    # dark bottle, light label: paper tone per column = bright percentile of the old label band (text/foil ignored)
    pb0, pb1 = int(a.yc - 0.40 * Y_H), int(a.yc + 0.40 * Y_H)
    band_px = RO[pb0:pb1]
    Lb = band_px.mean(2)
    thr = np.percentile(Lb, 60, axis=0)
    paper_col = np.zeros((W, 3), np.float32)
    for x in range(W):
        sel = band_px[:, x][Lb[:, x] >= thr[x]]
        paper_col[x] = np.median(sel, axis=0) if len(sel) else band_px[:, x].mean(0)
    paper_col = cv2.GaussianBlur(paper_col[None], (0, 0), 10)[0]
    if a.light_ink:   # dark label: keep the scene's soft shading but no new streaks / colour cast
        paper_col = cv2.GaussianBlur(paper_col[None], (0, 0), 45)[0]
        _m = paper_col[(np.arange(W) > XL + 8) & (np.arange(W) < XR - 8)].mean(0, keepdims=True)
        paper_col = _m + 0.3 * (paper_col - _m)
        _lum = paper_col.mean(1, keepdims=True); paper_col = _lum + 0.45 * (paper_col - _lum)
        paper_col = paper_col * 0.78
    PAPER = np.repeat(paper_col[None], H, 0)

# ---------- 2. artwork classes
_art = Image.open(a.art).convert('RGBA'); _bg = Image.new('RGBA', _art.size, (255, 255, 255, 255)); _bg.alpha_composite(_art)
# mockup exports often carry a 1-px light frame on the outer rows/cols -> it shows as a white line along the label edge; replace with the inner neighbour
_ba = np.array(_bg); _ba[:2] = _ba[2]; _ba[-2:] = _ba[-3]; _ba[:, :2] = _ba[:, 2:3]; _ba[:, -2:] = _ba[:, -3:-2]; _bg = Image.fromarray(_ba)
A = np.asarray(_bg.convert('RGB')).astype(np.float32)
for p in a.patch:
    dx, dy, sx, sy, w, h = map(int, p.split(','))
    A[dy:dy + h, dx:dx + w] = np.minimum(A[dy:dy + h, dx:dx + w], A[sy:sy + h, sx:sx + w])
ah, aw = A.shape[:2]
r, g, b = A[..., 0], A[..., 1], A[..., 2]
green = ((g - r > 50) & (g - b > 50)).astype(np.float32)
mag = ((r - g > 80) & (b - g > 40)).astype(np.float32)
neutral = (np.abs(r - g) < 12) & (np.abs(g - b) < 12)
dark = (neutral & (A.mean(2) < 200)).astype(np.uint8)
seed = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
_, lab = cv2.connectedComponents(dark, connectivity=8)          # whole grey shapes incl. sharp tips
keep = np.unique(lab[seed > 0]); keep = keep[keep > 0]
greyfill = cv2.dilate(np.isin(lab, keep).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
if a.light_ink: greyfill[:] = 0
ink = A.copy(); ink[green > 0] = 255; ink[mag > 0] = 255; ink[greyfill > 0] = 255

# ---------- 3. cylinder wrap
cx, Rr, k = (XL + XR) / 2, (XR - XL) / 2, a.k
th = np.arcsin(np.clip((xs - cx) / Rr, -0.999, 0.999))
sv = 1 - a.bulge * (1 - np.cos(th))
topm, botm = a.yc - ah / 2 * k * sv, a.yc + ah / 2 * k * sv
top, bot = a.yc - ah / 2 * k, a.yc + ah / 2 * k
U = (a.uc + th * Rr / k).astype(np.float32); V = ((ys - a.yc) / (k * sv) + ah / 2).astype(np.float32)
if a.top0 is not None:   # label edges follow measured perspective arcs
    _c = np.cos(th); topm = a.top0 - a.arc_top * (1 - _c); botm = a.bot0 - a.arc_bot * (1 - _c)
    if a.corner_slope:
        _u = np.abs((xs - cx) / Rr)
        def _rise(arc, s=a.corner_slope):
            us = s * Rr / np.sqrt(arc ** 2 + (s * Rr) ** 2)
            return arc * (1 - np.sqrt(1 - np.minimum(_u, us) ** 2)) + s * Rr * np.maximum(_u - us, 0)
        topm = a.top0 - _rise(a.arc_top); botm = a.bot0 - _rise(a.arc_bot)
    top, bot = a.top0, a.bot0
    V = ((ys - topm) / (botm - topm) * ah).astype(np.float32)
inside = cv2.GaussianBlur(((xs > XL + 1) & (xs < XR - 1) & (ys >= topm) & (ys <= botm)).astype(np.float32), (0, 0), 0.7)
rm = lambda img, it=cv2.INTER_CUBIC: cv2.remap(img, U, V, it, borderMode=cv2.BORDER_REPLICATE)
INK = rm(ink); GR = rm(green, cv2.INTER_LINEAR); MG = rm(mag, cv2.INTER_LINEAR); SF = rm(greyfill, cv2.INTER_LINEAR)

CL = (C if a.paper == 'body' else PAPER).mean(2); band = (ys > top) & (ys < bot)
np.savez_compressed(a.out.replace('.png', '_masks2.npz'), GR=GR.astype(np.float16), MG=MG.astype(np.float16), SF=SF.astype(np.float16), inside=inside.astype(np.float16))
var = CL / 200.0          # surface brightness relative to a white-bottle reference (approved look)

# ---------- 4. gold from the source: column brightness of the old gold
Ro, Bo = RO[..., 0], RO[..., 2]
og = (Ro - Bo > 40) & (Ro > 110) & (xs > XL + 4) & (xs < XR - 4) & (ys > Y0 + 28) & (ys < Y1 - 25)
prof = np.zeros((W, 3), np.float32); cnt = np.zeros(W, np.float32)
yy, xx = np.where(og); np.add.at(prof, xx, RO[yy, xx]); np.add.at(cnt, xx, 1)
xi = np.arange(W); good = cnt > 3
Lp = np.interp(xi, xi[good], (prof[good].mean(1) / cnt[good])) if good.any() else np.full(W, 180.0)
Lp = cv2.GaussianBlur(Lp[None].astype(np.float32), (0, 0), 5)[0]
inb = (xi > XL + 6) & (xi < XR - 6)
p5, p95 = np.percentile(Lp[inb], [5, 95]); tcol = np.clip((Lp - p5) / (p95 - p5 + 1e-3), 0, 1)
sm = lambda t: t * t * (3 - 2 * t)
sheen = 0.5 + 0.5 * np.cos((U * 0.0045 + V * 0.009) * 2 * np.pi)
tt = np.clip(sm(sm(tcol))[None] * 0.7 + 0.22 * sheen + ((np.clip(var * 2.5, 0.75, 1.0) if a.light_ink else var) - 1) * 2.0 + 0.02, 0, 1)
stops = np.array([[66, 52, 34], [128, 102, 66], [192, 163, 118], [236, 216, 178], [252, 245, 222]], np.float32)
pos = np.array([0, 0.3, 0.6, 0.85, 1.0])
gold = np.stack([np.interp(tt, pos, stops[:, c]) for c in range(3)], -1)

# ---------- 5. compose label
paper = C * 0.975 if a.paper == 'body' else PAPER
if a.paper == 'scene':
    C_in = PAPER   # surface under the label is the paper itself
if a.light_ink:
    tl = np.clip((INK.mean(2) / 255.0 - 0.5) / 0.5, 0, 1)[..., None]
    light = np.array([222, 219, 212], np.float32)[None, None] * np.clip(var[..., None] * 3.0, 0.6, 1.0)
    lab_ = paper * (1 - tl) + light * tl
else:
    lab_ = paper * (INK / 255.0)
sat = np.array([168, 168, 170], np.float32)[None, None] * (var[..., None] ** 1.2)
lab_ = lab_ * (1 - SF[..., None]) + sat * SF[..., None]
p50, p99 = np.percentile(CL[band & (xs > XL + 5) & (xs < XR - 5)], [50, 99])
glint = np.clip((CL - p50) / (p99 - p50 + 1e-3), 0, 1) ** 3
uv = paper * 0.955 + (glint * (38.0 if a.paper == 'body' else 16.0))[..., None]
lab_ = lab_ * (1 - MG[..., None]) + uv * MG[..., None]
lab_ = lab_ * (1 - GR[..., None]) + gold * GR[..., None]
edge = np.zeros((H, W), np.float32)
for yE, s in [(topm, 1), (botm, -1)]:
    d = (ys - yE) * s
    edge += -10 * np.exp(-((d + 1.5) / 1.2) ** 2) + a.edge_light * np.exp(-((d - 1.0) / 0.8) ** 2)
lab_ = cv2.GaussianBlur(lab_, (0, 0), 0.55)
res = C * (1 - inside[..., None]) + lab_ * inside[..., None]
res += (edge * ((xs > XL + 1) & (xs < XR - 1)))[..., None]
res = np.clip(res, 0, 255).astype(np.uint8)
Image.fromarray(res).save(a.out.replace('.png', '_upright.png'))

# ---------- 6. back to the scene, only the rebuilt band
Mb = np.zeros((H, W), np.float32); Mb[Y0 - 22:Y1 + 25, int(XL) + 1:int(XR) - 1] = 1
Mb = cv2.GaussianBlur(Mb, (0, 0), 1.0)
back = Image.fromarray(res).rotate(-a.angle, center=CEN, resample=Image.BICUBIC)
wm = Image.fromarray((Mb * 255).astype(np.uint8)).rotate(-a.angle, center=CEN, resample=Image.BILINEAR)
out = Image.composite(back, S_img, wm); out.save(a.out)
bb = wm.getbbox(); box = (bb[0] - 40, bb[1] - 40, bb[2] + 40, bb[3] + 40)
x0, y0 = S_img.crop(box), out.crop(box)
cmp_ = Image.new('RGB', (x0.width * 2 + 20, x0.height), 'white'); cmp_.paste(x0, (0, 0)); cmp_.paste(y0, (x0.width + 20, 0))
cmp_.save(a.out.replace('.png', '_compare.png'))
print('done', a.out, 'label rows', round(top, 1), round(bot, 1))
