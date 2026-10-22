# label_finish.py - second pass after label_rewrap.py (dark label): scene light, UV-varnish / foil gloss, paper grain,
# and a clip to the real bottle silhouette (per-row edges measured on the raw scene).
# usage: python3 label_finish.py scene.png rewrapped.png rewrapped_masks2.npz out.png --edges XL,XR \
#            --clean_rows y0,y1,y2,y3 --label_rows ya,yb --yc YC
#   scene.png         raw generation (same size as the rewrapped image)
#   rewrapped*.png    output of label_rewrap.py (it also writes <out>_masks2.npz)
#   --edges           bottle body left,right x at the label rows (same as label_rewrap --edges, widest measured)
#   --clean_rows      two clean strips of the body, above and below the label: y0,y1 (above), y2,y3 (below) - light profile source
#   --label_rows      rows searched for the silhouette edges (a little above / below the label)
#   --yc              label centre row (for the soft diagonal reflection)
import argparse, numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument('scene'); ap.add_argument('rw'); ap.add_argument('masks'); ap.add_argument('out')
ap.add_argument('--edges', required=True); ap.add_argument('--clean_rows', required=True)
ap.add_argument('--label_rows', required=True); ap.add_argument('--yc', type=float, required=True)
ap.add_argument('--body_rows', default=None, help='y0,y1: clean glass right under the label; label edge is pulled in to the line through the label top edge and this glass edge (no paper ledge)')
a_ = ap.parse_args()
scene, rw, masks, out = a_.scene, a_.rw, a_.masks, a_.out
XL, XR = map(float, a_.edges.split(','))
CR = list(map(int, a_.clean_rows.split(','))); LR = list(map(int, a_.label_rows.split(','))); YC = a_.yc
S = np.asarray(Image.open(scene).convert('RGB')).astype(np.float32)
R = np.asarray(Image.open(rw).convert('RGB')).astype(np.float32)
M = np.load(masks); GR, MG, ins = [M[k].astype(np.float32) for k in ('GR', 'MG', 'inside')]
H, W = R.shape[:2]; xs = np.arange(W, dtype=np.float32); ys = np.arange(H, dtype=np.float32)[:, None]
Lg = S.mean(2)
P = (Lg[CR[0]:CR[1]].mean(0) + Lg[CR[2]:CR[3]].mean(0)) / 2            # lighting of the clean body, per column
P = cv2.GaussianBlur(P[None], (0, 0), 6)[0]
inb = (xs > XL + 30) & (xs < XR - 30); med = np.median(P[inb])
Pn = P / med
cx, Rr = (XL + XR) / 2, (XR - XL) / 2
cth = np.sqrt(np.clip(1 - ((xs - cx) / Rr) ** 2, 0, 1))              # cos(theta) per column
Pn_c = np.where(xs > cx, np.minimum(Pn, 1.25), Pn)                      # right side only catches a weak bounce, never a bright strip
shade = (0.70 + 0.30 * np.clip(Pn_c, 0, 3.2)) * (0.55 + 0.45 * cth ** 0.6)   # satin paper under that light, darker toward the glass edges
spec = np.clip((P - 38) / 70, 0, 1) ** 1.4                           # where the body really catches the light
spec = cv2.GaussianBlur(spec[None], (0, 0), 5)[0]
sh2 = shade[None, :, None]
lab = R * sh2
# soft satin sheen on plain paper
lab += (spec * 16)[None, :, None]
# UV varnish: clear gloss, brighter where the body catches light, plus a long soft reflection
diag = np.exp(-(((xs[None, :] - (XL + 250)) + 0.28 * (ys - YC)) / 34.0) ** 2) * 0.55
uvl = (spec[None, :] * 150 + diag * 70) * MG
lab = lab * (1 - 0.10 * MG[..., None]) + uvl[..., None]
# gold foil: contrast and mirror-like highlights
gl = (spec[None, :] * 120 + diag * 55) * GR
lab = lab + gl[..., None] * np.array([1.0, 0.92, 0.70], np.float32)
# paper grain, fine and slightly fibrous
rng = np.random.default_rng(7)
n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8) * 3.0
nf = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=2.4) * 2.0
lab += (n + nf)[..., None] * (1 - 0.6 * GR[..., None])
lab = np.clip(lab, 0, 255)
# clip to the real bottle silhouette: near the two edges take the mask from the raw pixels (bottle is dark, backdrop and bag are light)
Ls = cv2.GaussianBlur(Lg, (0, 0), 0.8)
bgv = np.percentile(Lg[LR[0]:LR[1], int(XL) - 60:int(XL) - 30], 50)
_rb = cv2.GaussianBlur(S[..., 0] - S[..., 2], (0, 0), 0.8)          # warm beige (bag) vs neutral glass
sil_pix = np.clip((32 - _rb) / 14.0, 0, 1)   # only the warm bag is excluded by colour; backdrop is cut by the fitted edge line (the light glass rim must not eat the label corners)
sil = np.where((xs[None, :] > XL + 25) & (xs[None, :] < XR - 25), 1.0, sil_pix).astype(np.float32)
def _edge(y, lo, hi, sign):
    g = np.gradient(Ls[y, lo:hi]); i = int(np.argmin(g) if sign < 0 else np.argmax(g))
    if 0 < i < len(g) - 1:
        d = (g[i - 1] - g[i + 1]) / (2 * (g[i - 1] - 2 * g[i] + g[i + 1]) + 1e-9); return lo + i + float(d)
    return float(lo + i)
rr = np.arange(LR[0], LR[1], 5)
le = np.array([_edge(y, int(XL) - 45, int(XL) + 45, -1) for y in rr]); re_ = np.array([_edge(y, int(XR) - 45, int(XR) + 40, +1) for y in rr])
def _fit(v):
    p = np.polyfit(rr, v, 2)
    for _ in range(3):
        r = v - np.polyval(p, rr); k = np.abs(r) < 2.5; p = np.polyfit(rr[k], v[k], 2)
    return np.polyval(p, np.arange(H))
el, er = _fit(le)[:, None], _fit(re_)[:, None]
el_old, er_old = el.copy(), er.copy()
if a_.body_rows:
    B0, B1 = map(int, a_.body_rows.split(',')); rb = np.arange(B0, B1, 5); rt = np.arange(LR[0] + 100, LR[0] + 250, 5)   # upper part of the label (below the shoulder)
    def _line(fit, lo, hi, sign):
        bv = np.array([_edge(y, lo, hi, sign) for y in rb]); yy = np.r_[rt, rb]; vv = np.r_[fit[rt, 0], bv]
        p = np.polyfit(yy, vv, 1); return np.polyval(p, np.arange(H))[:, None]
    el = np.maximum(el, _line(el, int(XL) - 45, int(XL) + 45, -1)); er = np.minimum(er, _line(er, int(XR) - 45, int(XR) + 40, +1))
sil_fit = np.clip(xs[None, :] - el, 0, 1) * np.clip(er - xs[None, :], 0, 1)   # edge line from the gradient, 0.5 px inside
sil = sil * sil_fit                                                          # both must agree: nothing spills onto the bag or backdrop
sil = cv2.GaussianBlur(sil, (0, 0), 0.5)
a = (ins * sil)[..., None]
res = R * (1 - a) + lab * a
# outside the glass keep the raw pixels exactly
res = res * np.clip(sil, 0, 1)[..., None] + S * (1 - np.clip(sil, 0, 1))[..., None] * (ins > 0.02)[..., None] + R * (1 - (ins > 0.02))[..., None] * (1 - np.clip(sil, 0, 1))[..., None]
# away from the new label (beyond a 3-px seam) keep the raw scene: the erase band is only needed under the label
_insd = cv2.dilate((ins > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
_insd = (cv2.GaussianBlur(_insd, (0, 0), 1.0) * np.clip(sil, 0, 1))[..., None]   # and never outside the glass (label-edge bevel must not stick out)
res = res * _insd + S * (1 - _insd)
if a_.body_rows:
    # move the raw backdrop->glass transition in by the same sub-pixel shift (keeps the natural edge profile, no halo)
    _d = np.where(xs[None, :] < cx, el - el_old, er - er_old)
    _mx = (xs[None, :] - _d).astype(np.float32); _my = np.broadcast_to(ys, (H, W)).astype(np.float32)
    _sh = cv2.remap(S, _mx, np.ascontiguousarray(_my), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    _zone = (((xs[None, :] > el_old - 20) & (xs[None, :] < el + 2)) | ((xs[None, :] < er_old + 20) & (xs[None, :] > er - 2))).astype(np.float32)
    _zone[:LR[0]] = 0; _zone[LR[1]:] = 0
    _w = (_zone * (1 - sil_fit))[..., None]
    res = res * (1 - _w) + _sh * _w
Image.fromarray(np.clip(res, 0, 255).astype(np.uint8)).save(out)
print('saved', out, 'Pn range', round(float(Pn[inb].min()), 2), round(float(Pn[inb].max()), 2))
