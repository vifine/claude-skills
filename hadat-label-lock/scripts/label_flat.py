# label_flat.py - put a flat label mockup onto a label that sits on the (nearly flat) front face of a bottle,
# by a perspective quad, keeping the generated paper light, silver / gold modulation and the scene grain.
# For flat or oval bottles seen at an angle, where the cylinder wrap of label_rewrap.py does not fit.
# usage: python3 label_flat.py scene.png mockup.png out.png --angle A --pivot x,y --quad "TLx,TLy;TRx,TRy;BRx,BRy;BLx,BLy" [--only boxes]
#   --only           if the generated label is good, replace only the unreadable small print (boxes in mockup px)
#   --angle/--pivot  rotate the scene CCW by A about pivot so the bottle is upright (same convention as label_rewrap.py)
#   --quad           outer corners of the label in the UPRIGHT scene (measure on a zoomed grid)
#   mockup           flat label export from Figma (SKU - Label frame), any scale; white = paper, grey = print / silver, warm = gold
#   writes out.png and out_upright.png
import argparse, numpy as np, cv2
from PIL import Image
ap = argparse.ArgumentParser()
ap.add_argument('scene'); ap.add_argument('art'); ap.add_argument('out')
ap.add_argument('--angle', type=float, default=0.0); ap.add_argument('--pivot', default='0,0')
ap.add_argument('--quad', required=True)
ap.add_argument('--only', default=None, help='replace only these mockup boxes "x0,y0,x1,y1;..." (mockup px): keep the generated label, fix only unreadable small print')
ap.add_argument('--silver', type=float, default=1.45, help='silver foil brightness vs the mockup grey')
ap.add_argument('--gold', type=float, default=1.0, help='gold foil brightness vs the mockup gold')
ap.add_argument('--gold_sat', type=float, default=1.0, help='gold saturation (1 = as in the mockup)')
ap.add_argument('--ink_gamma', type=float, default=1.0)
ap.add_argument('--dark', action='store_true', help='dark label with light print (black bottles): print is laid over the dark paper, not multiplied')
ap.add_argument('--strip', type=float, default=20, help='--dark: how bright an old print stroke must be to be removed (lower = removes fainter old text, may smooth the paper grain)')
ap.add_argument('--light', type=float, default=1.0, help='--dark: brightness of light print vs the mockup')
ap.add_argument('--big', type=int, default=7, help='px (in scene) - grey shapes wider than this are silver foil, thinner ones are print')
a = ap.parse_args()
CEN = tuple(map(float, a.pivot.split(',')))
Q = np.float32([list(map(float, p.split(','))) for p in a.quad.split(';')])
S_img = Image.open(a.scene).convert('RGB')
U = np.asarray(S_img.rotate(a.angle, center=CEN, resample=Image.BICUBIC)).astype(np.float32)
H, W = U.shape[:2]
# mockup -> about 2x the target size, then warp
art = np.asarray(Image.open(a.art).convert('RGB')).astype(np.float32)
tw = 3 * max(np.linalg.norm(Q[1] - Q[0]), np.linalg.norm(Q[2] - Q[3])); sc = tw / art.shape[1]
art = cv2.resize(art, (int(art.shape[1] * sc), int(art.shape[0] * sc)), interpolation=cv2.INTER_AREA)
ah, aw = art.shape[:2]
Lm = art.mean(2); chroma = art[..., 0] - art[..., 2]
gold = np.clip((chroma - (38 if a.dark else 12)) / 18, 0, 1)   # dark labels print in warm taupe (chroma ~25) - that is print, not foil
grey = np.clip((235 - Lm) / 40, 0, 1) * (1 - gold)
k = max(3, int(round(a.big * 3)))                    # mockup is ~3x the scene
bigm = cv2.morphologyEx((grey > 0.5).astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8)).astype(np.float32)
bigm = cv2.dilate(bigm, np.ones((3, 3), np.uint8)).astype(np.float32) * grey
ink = grey * (1 - bigm)
Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [aw, 0], [aw, ah], [0, ah]]), Q)
bx0, by0 = np.floor(Q.min(0)).astype(int) - 4; bx1, by1 = np.ceil(Q.max(0)).astype(int) + 4
SS = 3                                                # supersampled warp, then area-downsample: thin print strokes stay whole
Tm = np.array([[SS, 0, -SS * bx0], [0, SS, -SS * by0], [0, 0, 1]], np.float64) @ Hm
def wp(x):
    big = cv2.warpPerspective(x, Tm, (SS * (bx1 - bx0), SS * (by1 - by0)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    small = cv2.resize(big, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W) + x.shape[2:], np.float32); out[by0:by1, bx0:bx1] = small; return out
A = wp(art); G = wp(gold); SB = wp(bigm); IK = wp(ink)
M = wp(np.ones((ah, aw), np.float32))                # label area (antialiased edge)
Min = cv2.erode((M > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
# paper light from the generated label: everything that is not plain paper (old print, old frame) is inpainted
Lg = U.mean(2)
loc = cv2.GaussianBlur(Lg, (0, 0), 6)
dirty = (((np.abs(Lg - loc) > 6) if a.dark else (Lg < loc - 6)) | (cv2.dilate(((SB + IK + G) > 0.15).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)).astype(np.uint8)
dirty = cv2.dilate(dirty, np.ones((5, 5), np.uint8)) * (M > 0.01).astype(np.uint8)
x0, y0 = np.floor(Q.min(0)).astype(int) - 12; x1, y1 = np.ceil(Q.max(0)).astype(int) + 12
crop = np.clip(U[y0:y1, x0:x1], 0, 255).astype(np.uint8)
P = U.copy()
P[y0:y1, x0:x1] = cv2.inpaint(crop, dirty[y0:y1, x0:x1], 7, cv2.INPAINT_TELEA).astype(np.float32)
P = P * 0.4 + cv2.GaussianBlur(P, (0, 0), 3) * 0.6
if a.dark:   # dark paper: remove the old light print by a grey opening, keep the paper grain and the scene highlights
    op = cv2.morphologyEx(np.clip(U, 0, 255).astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(np.float32)
    op = cv2.GaussianBlur(op, (0, 0), 1.0)
    txt = cv2.dilate((cv2.GaussianBlur(U.mean(2) - op.mean(2), (0, 0), 1.5) > a.strip).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)   # print strokes survive the blur, paper grain dots do not
    txt = cv2.GaussianBlur(txt, (0, 0), 0.8)[..., None]
    _w = 1 - txt                                     # opening also eats the bright paper grain: give its mean brightness back
    op = op + cv2.GaussianBlur((U - op) * _w, (0, 0), 3) / (cv2.GaussianBlur(_w[..., 0], (0, 0), 3)[..., None] + 1e-3)
    P = np.clip(U * (1 - txt) + op * txt, 0, 255)
# metal modulation: light / dark variation of the generated foil where the new foil lies (old foil sits there too)
mod = np.clip(Lg / (cv2.GaussianBlur(Lg, (0, 0), 14) + 1e-3), 0.75, 1.3)
mod = cv2.GaussianBlur(mod, (0, 0), 1.2)[..., None]
pl = P / 245.0
ink_c = (A / 255.0) ** a.ink_gamma                                              # >1 = darker print (readability at small size)
if a.dark:   # light print on dark paper: coverage from the mockup lightness, colour = mockup colour dimmed to scene level
    tl = np.clip((A.mean(2, keepdims=True) - 45) / 130.0, 0, 1) ** 0.75 * (1 - G[..., None])
    lab = P * (1 - tl) + np.maximum(P, A * a.light * (P.mean(2, keepdims=True) / 30.0) ** 0.15) * tl
    IK = IK * 0; SB = SB * 0
else:
    lab = P * (1 - IK[..., None]) + P * ink_c * IK[..., None]                   # print: multiply on paper
silver = A.mean(2, keepdims=True) * np.array([1.0, 0.99, 0.97], np.float32) * pl * a.silver * (1.0 + 1.2 * (mod - 1))   # neutral satin silver                            # satin silver: mockup grey lifted to foil brightness, scene light, foil variation
lab = lab * (1 - SB[..., None]) + silver * SB[..., None]
_al = A.mean(2, keepdims=True)
_gm = max(float(_al[G > 0.5].mean()) if (G > 0.5).any() else 128.0, 1.0)
champ = np.array([206, 184, 146], np.float32)                                     # champagne foil (HADAT), mockup gives only the light pattern
if a.dark:   # gold on a black label: own brightness, not multiplied by the dark paper
    gold_c = np.array([205, 178, 118], np.float32) * (_al / _gm) ** 0.9 * a.gold * (1.0 + 0.5 * (mod - 1))
else:
    gold_c = champ * (_al / _gm) ** 0.8 * pl * a.gold * (1.0 + 1.2 * (mod - 1))
_gl = gold_c.mean(2, keepdims=True); gold_c = _gl + a.gold_sat * (gold_c - _gl)
lab = lab * (1 - G[..., None]) + gold_c * G[..., None]
lab = cv2.GaussianBlur(lab, (0, 0), 0.45)
lab += np.random.default_rng(3).normal(0, 1.1, lab.shape).astype(np.float32)
if a.only:
    om = np.zeros((ah, aw), np.float32)
    for b in a.only.split(';'):
        bx = [int(round(float(v) * sc)) for v in b.split(',')]; om[bx[1]:bx[3], bx[0]:bx[2]] = 1
    om = cv2.GaussianBlur(om, (0, 0), 3 * sc + 0.5)
    M = M * wp(om)
res = U * (1 - M[..., None]) + lab * M[..., None]
res = np.clip(res, 0, 255).astype(np.uint8)
Image.fromarray(res).save(a.out.replace('.png', '_upright.png'))
# back to the scene: only the label area (+2 px)
mk = cv2.GaussianBlur(cv2.dilate((M > 0.01).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32), (0, 0), 0.8)
back = Image.fromarray(res).rotate(-a.angle, center=CEN, resample=Image.BICUBIC)
wm = Image.fromarray((mk * 255).astype(np.uint8)).rotate(-a.angle, center=CEN, resample=Image.BILINEAR)
Image.composite(back, S_img, wm).save(a.out)
print('done', a.out)
