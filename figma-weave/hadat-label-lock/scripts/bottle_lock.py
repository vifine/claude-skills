"""Pose-independent bottle lock: project an upright Hero0 bottle (on white) onto a bottle
in any scene (tilted / lying / leaning), as a surface of revolution.

Model: hero rows = height along the axis, hero columns = horizontal offset from the axis.
Scene = similarity(alpha, s, t) of a view where the axis is tipped toward the camera by beta,
so horizontal lines (label edges, bottom) become ellipse arcs.
Pose is fitted automatically: PCA init -> silhouette IoU -> edge alignment.

usage: python3 bottle_lock.py hero.png scene.png out.png --pt X,Y [--tr 0.55] [--beta B] [--noedge]
  --pt     a point inside the bottle in the scene (seeds the bottle mask)
  --tr     strength of scene relighting on the glass (0..1)
  --beta   fix beta (degrees) instead of fitting
Writes out.png plus out_dbg.jpg (fitted contour over scene) and out_pose.json.
"""
import sys, json, argparse, numpy as np, cv2
from PIL import Image
from scipy.optimize import minimize

ap = argparse.ArgumentParser()
ap.add_argument('hero'); ap.add_argument('scene'); ap.add_argument('out')
ap.add_argument('--axis', required=True, help='x_top,y_top,x_base,y_base in scene (cap top centre -> base centre)')
ap.add_argument('--tr', type=float, default=0.55)
ap.add_argument('--beta', type=float, default=None)
ap.add_argument('--hcut', type=float, default=0, help='hero rows above this stay from the scene (e.g. mirrored pump)')
ap.add_argument('--thr', type=float, default=None)
ap.add_argument('--pose', default=None, help='reuse a saved _pose.json (skip fitting)')
ap.add_argument('--no_base_shadow', action='store_true', help='do not move shadows beyond the base (standing bottle on a reflective floor)')
ap.add_argument('--rim', type=float, default=2.0, help='hero rim (scene px) refilled from the interior')
ap.add_argument('--ao', type=float, default=0.3, help='ambient-occlusion darkening right at the silhouette (kills light gaps)')
ap.add_argument('--light', action='store_true', help='light/white bottle (hero on white): mask by <248, bright old-bottle detection, relight the body')
ap.add_argument('--no_occ', action='store_true', help='nothing stands in front of the bottle')
ap.add_argument('--keepscene', type=float, default=0.0, help='mix of scene speculars on glass')
a = ap.parse_args()

H = np.array(Image.open(a.hero).convert('RGB')).astype(np.float32)
S = np.array(Image.open(a.scene).convert('RGB')).astype(np.float32)
hh, hw = H.shape[:2]; sh, sw = S.shape[:2]
lum = lambda x: x @ np.array([0.299, 0.587, 0.114], np.float32)

# ---------------- hero: mask, axis, profile
Lh = lum(H)
hm = (Lh < (248 if a.light else 170)).astype(np.uint8)
hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
if a.light:
    cn0, _ = cv2.findContours(hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(hm, cn0, -1, 1, -1)
n, lab, st, _ = cv2.connectedComponentsWithStats(hm)
hm = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
# fill holes per row (label highlights)
rows = np.where(hm.any(1))[0]; ytop, ybot = rows.min(), rows.max()
Lx = np.full(hh, np.nan); Rx = np.full(hh, np.nan)
for y in rows:
    xs = np.where(hm[y])[0]; Lx[y], Rx[y] = xs.min(), xs.max()
# refine each row's edges to the strongest luminance step (packshots have a soft grey shadow ramp
# next to the bottle that a plain threshold swallows)
Lhb = cv2.GaussianBlur(Lh, (0, 0), 0.8)
gxh = np.diff(Lhb, axis=1)
dL, dR = [], []
for y in rows:
    l, r = int(Lx[y]), int(Rx[y])
    a0, a1 = max(1, l - 4), min(hw - 2, l + 14)
    b0, b1 = max(1, r - 14), min(hw - 2, r + 4)
    if a1 > a0 and b1 > b0:
        dL.append(a0 + int(np.argmin(gxh[y, a0:a1])) + 1 - l); dR.append(r - (b0 + int(np.argmax(gxh[y, b0:b1]))))
offL = int(np.clip(np.median(dL), 0, 15)); offR = int(np.clip(np.median(dR), 0, 15))
Lx[rows] += offL; Rx[rows] -= offR
print('hero edge trim (shadow ramp) L/R px:', offL, offR)
w = Rx - Lx
body = (w > np.nanmax(w) * 0.9)
if a.light:
    yr_ = np.arange(hh); L0 = rows.max() - rows.min()
    body = (yr_ > rows.min() + 0.45 * L0) & (yr_ < rows.min() + 0.8 * L0) & np.isfinite(w)
cx0 = float(np.nanmedian(((Lx + Rx) / 2)[body]))
from scipy.ndimage import median_filter
rprof = np.where(np.isfinite(Lx), (Rx - Lx) / 2, 0).astype(np.float32)
rprof = median_filter(rprof, 31).astype(np.float32)
rprof = np.maximum(rprof, 1)
# symmetric, smooth silhouette (a surface of revolution): removes bumps such as label art that pokes
# past the limb in the packshot
from scipy.ndimage import gaussian_filter1d
rs = gaussian_filter1d(median_filter(np.where(np.isfinite(Lx), (Rx - Lx) / 2, 0).astype(np.float32), 41), 3)
if a.light:   # contact shadow on the white packshot widens the base rows: radius may only shrink below the widest body row
    L_ = rows.max() - rows.min(); mid = rows[(rows > rows.min() + 0.45 * L_) & (rows < rows.min() + 0.8 * L_)]
    rbody = float(np.median(rs[mid])); yb_ = int(rows.min() + 0.8 * L_)
    rs[yb_:] = np.minimum.accumulate(np.minimum(rs[yb_:], rbody))
    cx_body = float(np.median(((Lx + Rx) / 2)[mid]))
cxr = gaussian_filter1d(median_filter(np.where(np.isfinite(Lx), (Rx + Lx) / 2, cx0).astype(np.float32), 61), 5)
if a.light:
    rprof = np.minimum(rprof, np.maximum(rs, 1)).astype(np.float32)
    cxr[int(rows.min() + 0.8 * (rows.max() - rows.min())):] = cx_body
hm_fill = np.zeros_like(hm)
xx_h = np.arange(hw)[None, :]
hm_fill[rows] = (np.abs(xx_h - cxr[rows, None]) <= rs[rows, None]).astype(np.uint8)
# never wider than the real bottle (rounded base/shoulder corners would otherwise pull in grey backdrop)
real = (Lh < (248 if a.light else 150)).astype(np.uint8)
real = cv2.morphologyEx(real, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
cn_h, _ = cv2.findContours(real, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); real_f = np.zeros_like(real); cv2.drawContours(real_f, cn_h, -1, 1, -1)
hm_fill &= cv2.erode(real_f, np.ones((3, 3), np.uint8))
# keep the raw outline where the shape is not rotational (spouts etc. above the cut)
for y in rows:
    if y < getattr(a, 'hcut', 0): hm_fill[y, int(Lx[y]):int(Rx[y]) + 1] = 1
# bottom edge per column: the grey contact shadow of the packshot sits right under the base and a
# threshold mask swallows it -> cut at the strongest dark->bright step going down
gyh = np.diff(Lhb, axis=0)
cols = np.where(hm_fill.any(0))[0]
bot_edge = np.full(hw, -1, np.int32)
for x in cols:
    ysx = np.where(hm_fill[:, x])[0]
    yb = ysx.max(); a0 = max(ytop, yb - (110 if a.light else 30))
    seg = gyh[a0:yb + 3, x]
    if seg.size: bot_edge[x] = a0 + int(np.argmax(seg))
ok = bot_edge >= 0
bot_s = bot_edge.astype(np.float32)
bot_s[ok] = median_filter(bot_s[ok], 15)
for x in cols:
    if bot_edge[x] >= 0: hm_fill[bot_s[x].astype(int) + 1:, x] = 0
hm = hm_fill
ybot = int(np.where(hm.any(1))[0].max())
center = (np.abs(np.arange(hw) - cx0) < 0.3 * rs.max()) & ok
print('hero base cut: rows trimmed (centre) ~', int(np.median(np.array([np.where(hm_fill[:, x])[0].max() for x in np.where(center)[0]]) - 0)) if center.any() else '-')
yref = float(ybot)
hero_len = ybot - ytop
print('hero axis cx0=%.1f rows %d-%d rmax=%.1f' % (cx0, ytop, ybot, rprof.max()))

# ---------------- scene inputs
Ls = lum(S)
sat = S.max(-1) - S.min(-1); R_, G_, B_ = S[..., 0], S[..., 1], S[..., 2]
# occluders in front: warm saturated objects (tortoise clip etc.)
occ = ((R_ > G_ * 1.25) & (R_ > B_ * 1.6) & (sat > 55) & (Ls > 35)).astype(np.uint8)
occ = cv2.morphologyEx(occ, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
if a.no_occ: occ[:] = 0

# hero axis key points: top of the bottle on the axis (cap/actuator top) and base centre
col = hm[:, int(round(cx0)) - 3:int(round(cx0)) + 4].max(1)
htop = float(np.where(col > 0)[0].min())

# ---------------- projection
def solve(p, X, Y):
    """scene px -> hero px. p = alpha(deg), s, tx, ty, beta(deg); (tx,ty) = scene pos of hero base centre"""
    al, s, tx, ty, be = p[:5]; k = p[5] if len(p) > 5 else 0.0
    ca, sa = np.cos(np.radians(al)), np.sin(np.radians(al))
    dx, dy = (X - tx) / s, (Y - ty) / s
    Xp = ca * dx + sa * dy; Yp = -sa * dx + ca * dy          # rotated, hero-scale
    cb, sb = np.cos(np.radians(be)), np.sin(np.radians(be))
    h = yref + Yp / cb
    for _ in range(6):
        hi = np.clip(h, 0, hh - 1)
        r = np.interp(hi, np.arange(hh), rprof)
        Xh = Xp / (1 + k * (yref - hi) / hero_len)          # perspective taper along the axis
        ct = np.sqrt(np.clip(1 - (Xh / r) ** 2, 0, 1))
        h = yref + (Yp - r * ct * sb) / cb
    return (cx0 + Xh).astype(np.float32), h.astype(np.float32)

def render(p, img, shape, step=1, interp=cv2.INTER_LINEAR):
    ys, xs = np.mgrid[0:shape[0]:step, 0:shape[1]:step].astype(np.float32)
    mx, my = solve(p, xs, ys)
    return cv2.remap(img, mx, my, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# ---------------- init from two scene points: axis top -> base centre
x1_, y1_, x2_, y2_ = map(float, a.axis.split(','))
v = np.array([x2_ - x1_, y2_ - y1_]); Lv = np.linalg.norm(v); v /= Lv
al0 = np.degrees(np.arctan2(-v[0], v[1]))
s0 = Lv / (yref - htop)
p0 = np.array([al0, s0, x2_, y2_, a.beta if a.beta is not None else 0.0, 0.0])
print('init', np.round(p0, 3))

hmf = hm.astype(np.float32)
m0 = render(p0, hmf, (sh, sw), 1) > 0.5
ys_, xs_ = np.where(m0)
mg = int(0.12 * s0 * rprof.max() * 2)
bx0, by0 = max(0, xs_.min() - mg), max(0, ys_.min() - mg); bx1, by1 = min(sw, xs_.max() + mg), min(sh, ys_.max() + mg)
roi = cv2.dilate(m0.astype(np.uint8), np.ones((2 * mg + 1, 2 * mg + 1), np.uint8)) * (1 - cv2.dilate(occ, np.ones((9, 9), np.uint8)))

gmag = lambda im: np.sqrt(cv2.magnitude(cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)))
def feats(Limg, sig):
    Lb = cv2.GaussianBlur(Limg, (0, 0), sig)
    return gmag(Lb), 255 - Lb
def ncc(A, B):
    A = A - A.mean(); B = B - B.mean()
    return (A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-6)

def remap1d(img, mx, my, border, val=0):
    n = mx.size; W_ = 4096; pad = (-n) % W_
    X = np.r_[mx, np.zeros(pad, np.float32)].reshape(-1, W_); Y = np.r_[my, np.zeros(pad, np.float32)].reshape(-1, W_)
    return cv2.remap(img, X, Y, cv2.INTER_LINEAR, borderMode=border, borderValue=val).ravel()[:n]

def fit(p, ds, sig, free_beta, direc):
    Es, Ds = feats(Ls, sig * s0); Eh, Dh = feats(Lh, sig)
    if a.light:   # white bottle: brighter than the scene, darker than the white packshot -> compare silhouettes
        Ds = cv2.GaussianBlur(Ls, (0, 0), sig * s0); Dh = cv2.GaussianBlur(hm.astype(np.float32) * 255, (0, 0), sig)
    gys, gxs = np.mgrid[by0:by1:ds, bx0:bx1:ds].astype(np.float32)
    R_ = roi[by0:by1:ds, bx0:bx1:ds] > 0
    Esc, Dsc = Es[by0:by1:ds, bx0:bx1:ds][R_], Ds[by0:by1:ds, bx0:bx1:ds][R_]
    gx_, gy_ = gxs[R_], gys[R_]
    def cost(z):
        pp = p.copy(); pp[0] += z[0]; pp[1] *= (1 + z[1]); pp[2] += z[2]; pp[3] += z[3]; pp[5] = z[5]
        if free_beta: pp[4] = z[4]
        mx, my = solve(pp, gx_, gy_)
        e = remap1d(Eh, mx, my, cv2.BORDER_REPLICATE)
        d = remap1d(Dh, mx, my, cv2.BORDER_CONSTANT, 0)
        v = my >= a.hcut
        return -(ncc(e[v], Esc[v]) + ncc(d[v], Dsc[v]))
    z0 = np.array([0, 0, 0, 0, p[4], p[5]], float)
    r = minimize(cost, z0, method='Powell', options=dict(xtol=1e-4, ftol=1e-6, maxiter=4000, direc=np.diag(direc)))
    z = r.x; pp = p.copy(); pp[0] += z[0]; pp[1] *= (1 + z[1]); pp[2] += z[2]; pp[3] += z[3]; pp[5] = z[5]
    if free_beta: pp[4] = z[4]
    return pp, -r.fun

if a.pose:
    jp = json.load(open(a.pose)); p = np.array([jp['alpha'], jp['s'], jp['tx'], jp['ty'], jp['beta'], jp.get('taper', 0.0)])
    if 'yref' in jp and abs(jp['yref'] - yref) > 0.5:      # hero base moved (edge trim): keep the same placement
        al_ = np.radians(p[0]); dv = (yref - jp['yref']) * p[1]
        p[2] += -np.sin(al_) * dv; p[3] += np.cos(al_) * dv
    print('pose loaded', np.round(p, 3))
else:
    p = p0.copy()
    fb = a.beta is None
    p, sc = fit(p, 8, 6, False, [2, 0.03, 12, 12, 1, 0.03]); print('fit coarse', np.round(p, 3), round(sc, 4))
    p, sc = fit(p, 4, 3, fb, [0.5, 0.01, 4, 4, 3, 0.03]); print('fit mid   ', np.round(p, 3), round(sc, 4))
    p, sc = fit(p, 2, 1.5, fb, [0.2, 0.003, 1.5, 1.5, 1, 0.01]); print('fit fine  ', np.round(p, 3), round(sc, 4))

# ---------------- hero edge decontamination: the outer rim of a packshot on white carries white
# antialiasing / rim light; refill it from the interior and use a clean soft alpha instead
rim = max(3, int(round(a.rim / max(p[1], 0.2))))
inner = cv2.erode(hm, np.ones((2 * rim + 1, 2 * rim + 1), np.uint8))
fillm = ((cv2.dilate(hm, np.ones((2 * rim + 9, 2 * rim + 9), np.uint8)) > 0) & (inner == 0)).astype(np.uint8)
Hc = cv2.inpaint(np.clip(H, 0, 255).astype(np.uint8), fillm, 5, cv2.INPAINT_TELEA).astype(np.float32)
Hc = np.where(inner[..., None] > 0, H, Hc)
ah = cv2.GaussianBlur(cv2.erode(hm, np.ones((3, 3), np.uint8)).astype(np.float32), (0, 0), max(0.6, 0.8 / max(p[1], 0.2)))

# ---------------- render full res
W = render(p, Hc, (sh, sw), 1, cv2.INTER_LANCZOS4)
M = render(p, hmf, (sh, sw), 1)
A = np.clip(render(p, ah, (sh, sw), 1), 0, 1)
ys_, xs_ = np.mgrid[0:sh, 0:sw].astype(np.float32)
_, hgt = solve(p, xs_, ys_)
Mb = ((M > 0.5) & (hgt >= a.hcut)).astype(np.uint8)
if a.hcut > 0:   # soft seam where hero takes over from the scene
    seam = np.clip((hgt - a.hcut) / 6.0, 0, 1)
else:
    seam = np.ones_like(hgt)

# relight: robust shading ratio on a (hero-x, hero-h) grid -> follows cylinder lighting, ignores label art
Lw = lum(W)
hx, _ = solve(p, xs_, ys_)
inm = (Mb > 0) & (occ == 0)
nbx, nby = 16, 10
xr = hx[inm]; yr = hgt[inm]
x_edges = np.linspace(xr.min(), xr.max() + 1e-3, nbx + 1); y_edges = np.linspace(max(yr.min(), a.hcut), yr.max() + 1e-3, nby + 1)
bi = np.clip(np.digitize(hx, x_edges) - 1, 0, nbx - 1); bj = np.clip(np.digitize(hgt, y_edges) - 1, 0, nby - 1)
Rg = np.ones((nby, nbx, 3), np.float32); Cnt = np.zeros((nby, nbx))
for j in range(nby):
    for i in range(nbx):
        c = inm & (bi == i) & (bj == j)
        n_ = c.sum(); Cnt[j, i] = n_
        if n_ < 50: continue
        ls, lw = np.median(Ls[c]), np.median(Lw[c])
        r_ = np.clip((ls + 6) / (lw + 6), 0.35, 2.0)
        tt = [np.clip((np.median(S[..., k][c]) + 6) / (np.median(W[..., k][c]) + 6) / r_, 0.9, 1.12) for k in range(3)]
        Rg[j, i] = [r_ * t for t in tt]
# fill empty bins with nearest valid, smooth
valid = Cnt >= 50
if valid.any():
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    Rg = Rg[idx[0], idx[1]]
Rg = cv2.GaussianBlur(Rg, (0, 0), 0.8)
fx = ((hx - x_edges[0]) / (x_edges[-1] - x_edges[0]) * nbx - 0.5).astype(np.float32)
fy = ((hgt - y_edges[0]) / (y_edges[-1] - y_edges[0]) * nby - 0.5).astype(np.float32)
Gm = cv2.remap(Rg, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
wb = np.clip((Lw - 90) / 80, 0, 1)[..., None]   # foil / print highlights keep their brightness
if a.light:   # white body takes the scene light; only gold foil / dark print keep their own tone
    satw = W.max(-1) - W.min(-1)
    wb = np.maximum(np.clip((satw - 18) / 25, 0, 1), np.clip((150 - Lw) / 60, 0, 1))[..., None]
# ...except the rounded base: its bright rim is the white studio floor, it must take the scene's light
rows_b = np.where((np.arange(hh) > ytop + 0.85 * (ybot - ytop)) & (rs < 0.97 * rs.max()) & (rs > 0))[0]
if rows_b.size:
    yb0 = rows_b.min() - 4
    wb = wb * (hgt < yb0)[..., None]
G = 1 + (Gm - 1) * a.tr * (1 - 0.7 * wb)
Wl = np.clip(W * G, 0, 255)
glass = inm
if a.keepscene > 0:   # scene specular highlights on glass (reflections of the environment)
    hiS = np.clip(Ls - cv2.GaussianBlur(Ls, (0, 0), 6), 0, None)
    Wl = np.clip(Wl + (a.keepscene * hiS * (glass))[..., None], 0, 255)

# occluders: keep scene pixels of occluders that also exist outside the bottle
n, lab, st, _ = cv2.connectedComponentsWithStats(occ)
keep = np.zeros_like(occ)
outside = (Mb == 0)
for i in range(1, n):
    comp = lab == i
    if st[i, cv2.CC_STAT_AREA] > 300 and (comp & outside).sum() > 0.15 * st[i, cv2.CC_STAT_AREA]:
        keep |= comp.astype(np.uint8)
# grow occluders into the bottle with GrabCut (dark parts of the clip in front of dark glass)
if keep.any() and (keep & cv2.dilate(Mb, np.ones((9, 9), np.uint8))).any():
    near = cv2.dilate(keep, np.ones((61, 61), np.uint8)) > 0
    yy, xx = np.where(cv2.dilate(keep, np.ones((121, 121), np.uint8)) & cv2.dilate(Mb, np.ones((121, 121), np.uint8)))
    gx0, gx1, gy0, gy1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    gm_ = np.full((gy1 - gy0, gx1 - gx0), cv2.GC_PR_BGD, np.uint8)
    gm_[near[gy0:gy1, gx0:gx1]] = cv2.GC_PR_FGD
    gm_[cv2.erode(keep, np.ones((3, 3), np.uint8))[gy0:gy1, gx0:gx1] > 0] = cv2.GC_FGD
    gm_[~(cv2.dilate(keep, np.ones((121, 121), np.uint8)) > 0)[gy0:gy1, gx0:gx1]] = cv2.GC_BGD
    try:
        cv2.grabCut(cv2.cvtColor(S[gy0:gy1, gx0:gx1].astype(np.uint8), cv2.COLOR_RGB2BGR), gm_, None,
                    np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
        fg = ((gm_ == 1) | (gm_ == 3)).astype(np.uint8)
        Sc = S[gy0:gy1, gx0:gx1]; satc = Sc.max(-1) - Sc.min(-1); Lc = lum(Sc)
        cliplike = (satc > 45) | ((Lc < 70) & (Sc[..., 0] > Sc[..., 2] * 1.2))     # never grab plain background
        fg = (fg & cliplike).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        keep[gy0:gy1, gx0:gx1] |= fg
    except Exception as e:
        print('grabcut skip', e)
keep = cv2.dilate(keep, np.ones((3, 3), np.uint8))

# ---------------- old (generated) bottle silhouette via GrabCut, seeded by the new silhouette
yy, xx = np.where(cv2.dilate(Mb, np.ones((161, 161), np.uint8)) > 0)
gx0, gx1, gy0, gy1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
gm_ = np.full((gy1 - gy0, gx1 - gx0), cv2.GC_BGD, np.uint8)
crop_ = lambda m: m[gy0:gy1, gx0:gx1]
band_out = cv2.dilate(Mb, np.ones((71, 71), np.uint8)) > 0
gm_[crop_(band_out)] = cv2.GC_PR_BGD
gm_[crop_(Mb > 0)] = cv2.GC_PR_FGD
gm_[crop_(cv2.erode(Mb, np.ones((31, 31), np.uint8)) > 0)] = cv2.GC_FGD
gm_[crop_(keep > 0)] = cv2.GC_BGD
cv2.grabCut(cv2.cvtColor(crop_(S).astype(np.uint8), cv2.COLOR_RGB2BGR), gm_, None, np.zeros((1, 65)), np.zeros((1, 65)), 6, cv2.GC_INIT_WITH_MASK)
oldM = np.zeros_like(Mb)
oldM[gy0:gy1, gx0:gx1] = ((gm_ == 1) | (gm_ == 3)).astype(np.uint8)
oldM = cv2.morphologyEx(oldM, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
oldM = cv2.morphologyEx(oldM, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
# the generated bottle is much darker than its own contact shadow: keep only clearly dark pixels
in_med = np.median(Ls[(cv2.erode(Mb, np.ones((15, 15), np.uint8)) > 0) & ((Ls > np.percentile(Ls[Mb > 0], 50)) if a.light else (Ls < np.percentile(Ls[Mb > 0], 50)))])
ring2 = (cv2.dilate(Mb, np.ones((2 * int(0.3 * p[1] * 2 * rprof.max()) + 1,) * 2, np.uint8)) > 0) & (cv2.dilate(Mb, np.ones((41, 41), np.uint8)) == 0) & (keep == 0)
bg_med = np.median(Ls[ring2])
thr_o = in_med + 0.45 * (bg_med - in_med)
oldM = (oldM & ((Ls > thr_o) if a.light else (Ls < thr_o)) | (oldM & (Mb > 0))).astype(np.uint8)
oldM = cv2.morphologyEx(oldM, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
print('old bottle thr', round(float(thr_o), 1), 'in', round(float(in_med), 1), 'bg', round(float(bg_med), 1))
n, lab, st, _ = cv2.connectedComponentsWithStats(oldM)
if n > 1: oldM = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
# close notches where bright label art of the generated bottle touched its edge, then fill holes
kc = 2 * max(3, int(round(0.03 * p[1] * 2 * rprof.max()))) + 1
oldM = cv2.morphologyEx(oldM, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
cn_, _ = cv2.findContours(oldM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(oldM, cn_, -1, 1, -1)
oldM = (oldM & ~((keep > 0))).astype(np.uint8)
# lit parts of the generated bottle (grey collar, glossy rims) are not dark: add GrabCut pixels near the
# dark core that clearly differ from the local background (estimated by inpainting the bottle away)
gc_fg = np.zeros_like(Mb); gc_fg[gy0:gy1, gx0:gx1] = ((gm_ == 1) | (gm_ == 3))
near_o = cv2.dilate(oldM, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
yy_, xx_ = np.where(near_o); bx0_, bx1_, by0_, by1_ = max(0, xx_.min() - 20), min(sw, xx_.max() + 20), max(0, yy_.min() - 20), min(sh, yy_.max() + 20)
hole_ = (near_o | (Mb > 0))[by0_:by1_, bx0_:bx1_].astype(np.uint8)
bgE = cv2.inpaint(S[by0_:by1_, bx0_:bx1_].astype(np.uint8), hole_, 7, cv2.INPAINT_TELEA).astype(np.float32)
bgE = cv2.GaussianBlur(bgE, (0, 0), 3)
dif = np.zeros(Mb.shape, np.float32); dif[by0_:by1_, bx0_:bx1_] = np.sqrt(((S[by0_:by1_, bx0_:bx1_] - bgE) ** 2).sum(-1))
lit = (gc_fg > 0) & near_o & (dif > 38) & (keep == 0)
oldM = (oldM | lit).astype(np.uint8)
oldM = cv2.morphologyEx(oldM, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
dil_ = 2 * max(2, int(round(0.008 * p[1] * 2 * rprof.max()))) + 1
oldM = cv2.dilate(oldM, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_, dil_)))   # include the soft rim of the old bottle
base_lim = (yref - 4) if a.no_base_shadow else (yref + 0.15 * (yref - htop))
sl = (oldM > 0) & (cv2.dilate(Mb, np.ones((3, 3), np.uint8)) == 0) & (keep == 0) & (hgt < base_lim) & (hgt > a.hcut + 6)
sl = cv2.dilate(sl.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
sl = (cv2.GaussianBlur(sl.astype(np.float32), (0, 0), 3.0) > 0.25) & ~((Mb > 0) & (A > 0.98))   # smooth outline, no stair-steps
sl &= (keep == 0)

# ---------------- mirror-fill the slivers with background texture from just outside the old silhouette
base = S.copy()
if sl.any():
    # translate a strip of background texture into each sliver, perpendicular to the bottle axis
    # (along the axis at the ends) - keeps the fabric/paper grain intact, no smearing
    D = cv2.distanceTransform(cv2.dilate(oldM, np.ones((5, 5), np.uint8)), cv2.DIST_L2, 5)
    al = np.radians(p[0]); ax_dn = np.array([-np.sin(al), np.cos(al)])     # hero +y (towards base) in scene
    ax_rt = np.array([np.cos(al), np.sin(al)])                               # hero +x in scene
    bad = (cv2.dilate(oldM, np.ones((7, 7), np.uint8)) > 0) | (Mb > 0) | (keep > 0)
    src = S.copy(); done = ~sl
    n_, lab_, st_, _ = cv2.connectedComponentsWithStats(sl.astype(np.uint8))
    hl = yref - htop
    for i in range(1, n_):
        c = lab_ == i
        hm_c = np.median(hgt[c]); xm_c = np.median(hx[c])
        if hm_c > yref - 0.06 * hl: dvec = ax_dn
        elif hm_c < max(a.hcut, htop) + 0.04 * hl: dvec = -ax_dn
        else: dvec = ax_rt if xm_c > cx0 else -ax_rt
        th = float(D[c].max())
        cy, cx = np.where(c)
        for mult in (2.0, 2.8, 3.8, 5.0, 6.5):
            sh_ = mult * th + 6
            sx = np.clip(np.round(cx + dvec[0] * sh_).astype(int), 0, sw - 1); sy = np.clip(np.round(cy + dvec[1] * sh_).astype(int), 0, sh - 1)
            if bad[sy, sx].mean() < 0.02:
                break
        mxp = (cx + dvec[0] * sh_).astype(np.float32); myp = (cy + dvec[1] * sh_).astype(np.float32)
        vals = np.stack([cv2.remap(S[..., k], mxp[None], myp[None], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)[0] for k in range(3)], -1)
        src[cy, cx] = vals; done[cy, cx] = True
    if (~done).any():
        src = cv2.inpaint(src.astype(np.uint8), (~done).astype(np.uint8), 7, cv2.INPAINT_TELEA).astype(np.float32)
    # keep the translated texture (high frequencies) but take the tone gradient from a smooth fill
    hole = cv2.dilate(sl.astype(np.uint8), np.ones((3, 3), np.uint8))
    ext = (cv2.dilate(sl.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~sl & (Mb == 0) & (keep == 0)
    near_ = cv2.dilate(sl.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    hole = (hole > 0) | (near_ & ((Mb > 0) | (oldM > 0)))            # never pull tone from the bottles themselves
    smooth = cv2.inpaint(S.astype(np.uint8), hole.astype(np.uint8), 9, cv2.INPAINT_TELEA).astype(np.float32)
    sgm = 3.0
    hf = src - cv2.GaussianBlur(src, (0, 0), sgm)
    src = np.where(sl[..., None], cv2.GaussianBlur(smooth, (0, 0), sgm) + hf, src)
    src = np.where(ext[..., None], S, src)
    fe = max(1.5, 0.006 * p[1] * 2 * rprof.max())
    slg = cv2.GaussianBlur(sl.astype(np.float32), (0, 0), fe)
    w_ = np.clip(slg * 2.0, 0, 1)[..., None]                          # sliver fully replaced, soft fade outside
    base = S * (1 - w_) + src * w_
print('sliver px', int(sl.sum()), 'occluder px', int(keep.sum()))

# ---------------- move the contact / ambient shadow from the old silhouette to the new one
diam = p[1] * 2 * rprof.max()
Dmax = int(max(12, 0.13 * diam))
hx_all = hx if 'hx' in dir() else solve(p, xs_, ys_)[0]
nkeys_h = 14
def keys_for(mask):
    # nearest boundary pixel of mask for every pixel outside -> (side, h-bin) of that boundary pixel
    inv = (mask == 0).astype(np.uint8)
    d, lbl = cv2.distanceTransformWithLabels(inv, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.where(inv == 0)                   # label i -> i-th zero pixel in row-major order
    ly = np.zeros(lbl.max() + 1, np.int32); lx = np.zeros(lbl.max() + 1, np.int32)
    lab_at = lbl[zy, zx]; ly[lab_at] = zy; lx[lab_at] = zx
    by, bx = ly[lbl], lx[lbl]
    hb = hgt[by, bx]; side = (hx_all[by, bx] > cx0).astype(np.int32)
    hbin = np.clip(((hb - max(a.hcut, htop)) / (yref - max(a.hcut, htop)) * nkeys_h).astype(np.int32), -1, nkeys_h)
    key = (hbin + 1) * 2 + side                   # 0..(nkeys_h+2)*2-1
    return d, key
nk = (nkeys_h + 2) * 2
d_old, k_old = keys_for(oldM)
d_new, k_new = keys_for(Mb)
valid_bg = (keep == 0) & (Mb == 0) & (oldM == 0)
# parametric shadow profile per key: r(d) = 1 - A*exp(-d/tau)  (smooth -> no iso-lines / ghost contours)
dgrid = np.arange(Dmax + 1, dtype=np.float32)
Ak = np.zeros(nk, np.float32); Tk = np.full(nk, Dmax / 5, np.float32); Wk = np.zeros(nk, np.float32)
for kk in range(nk):
    selk = valid_bg & (k_old == kk)
    far = selk & (d_old > Dmax) & (d_old <= 1.4 * Dmax)
    if far.sum() < 30: continue
    ref = np.median(Ls[far])
    sel2 = selk & (d_old >= 1) & (d_old <= Dmax)
    dd = np.round(d_old[sel2]).astype(int); vv = Ls[sel2] / max(ref, 1)
    prof = np.full(Dmax + 1, np.nan, np.float32)
    for di in range(1, Dmax + 1):
        v_ = vv[dd == di]
        if v_.size >= 5: prof[di] = np.median(v_)
    ok = np.isfinite(prof)
    if ok.sum() < Dmax // 3: continue
    best = None
    for tau in np.linspace(max(1.5, Dmax / 15), Dmax / 2, 24):
        e = np.exp(-dgrid[ok] / tau); y = 1 - np.clip(prof[ok], 0, 1.2)
        A_ = float(np.clip((e * y).sum() / max((e * e).sum(), 1e-6), 0, 0.65))
        err = ((y - A_ * e) ** 2).sum()
        if best is None or err < best[0]: best = (err, A_, tau)
    Ak[kk], Tk[kk], Wk[kk] = best[1], best[2], 1.0
# smooth parameters along the bottle (same side), fill keys without data
for side in (0, 1):
    idx = np.arange(side, nk, 2)
    w = Wk[idx]
    if w.sum() == 0: continue
    g = lambda v: np.convolve(v, np.array([1, 2, 3, 2, 1], np.float32), 'same')
    Ak[idx] = g(Ak[idx] * w) / np.maximum(g(w), 1e-3)
    Tk[idx] = g(Tk[idx] * w) / np.maximum(g(w), 1e-3)
    Tk[idx] = np.where(g(w) > 0, Tk[idx], Dmax / 5)
R = (1 - Ak[:, None] * np.exp(-dgrid[None, :] / np.maximum(Tk[:, None], 1))).astype(np.float32)
R[:, -1] = 1.0
def lookup(d, k):
    di = np.clip(d, 0, Dmax).astype(np.float32)
    return cv2.remap(R, di, k.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
# same key for old and new distance -> where the outline did not move the gain is exactly 1
r_old = np.where(d_old < Dmax, lookup(np.maximum(d_old, 0.5), k_new), 1.0)
r_new = np.where(d_new < Dmax, lookup(np.maximum(d_new, 0.5), k_new), 1.0)
print('shadow A', np.round(Ak[Wk > 0], 2).tolist()[:8], 'tau', np.round(Tk[Wk > 0], 1).tolist()[:8])
mod = (keep == 0) & (M <= 0.5) & ~((oldM > 0) & (hgt <= a.hcut + 6))
mod = cv2.erode(mod.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
if a.no_base_shadow:
    mod &= (hgt < yref - 4)
# pixels that were the old bottle (now filled) had no old shadow
r_old = np.where(oldM > 0, 1.0, r_old)
gain = np.where(mod, np.minimum(r_new / np.maximum(r_old, 0.3), 1.0), 1.0).astype(np.float32)   # only add shadow where the bottle grew; never lighten
# smooth across the Voronoi seams of the nearest-boundary lookup (normalized within the background)
mf = mod.astype(np.float32); sg = max(2.0, Dmax / 7)
gain = np.where(mod, cv2.GaussianBlur(gain * mf, (0, 0), sg) / np.maximum(cv2.GaussianBlur(mf, (0, 0), sg), 1e-3), 1.0).astype(np.float32)
base = np.clip(base * gain[..., None], 0, 255)
print('shadow Dmax', Dmax, 'gain range', round(float(gain.min()), 3), round(float(gain.max()), 3))

# final tone match of the filled slivers to their immediate (already re-shadowed) surroundings
if sl.any():
    near2 = cv2.dilate(sl.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0
    hole2 = (sl | (near2 & ((Mb > 0) | (M > 0.3)))).astype(np.uint8)
    yy2, xx2 = np.where(near2); q0, q1, r0, r1 = max(0, yy2.min() - 10), min(sh, yy2.max() + 10), max(0, xx2.min() - 10), min(sw, xx2.max() + 10)
    sm2 = cv2.inpaint(np.clip(base[q0:q1, r0:r1], 0, 255).astype(np.uint8), hole2[q0:q1, r0:r1], 5, cv2.INPAINT_TELEA).astype(np.float32)
    sm2 = cv2.GaussianBlur(sm2, (0, 0), 2.5)
    bc = base[q0:q1, r0:r1]
    hf2 = bc - cv2.GaussianBlur(bc, (0, 0), 2.5)
    slc = sl[q0:q1, r0:r1]
    wq = cv2.GaussianBlur(slc.astype(np.float32), (0, 0), 1.2)[..., None]
    base[q0:q1, r0:r1] = bc * (1 - wq) + (sm2 + hf2) * wq
# ambient occlusion: surfaces touching the bottle are always a little darker right at the edge
if a.ao > 0:
    tau = max(2.5, 0.012 * p[1] * 2 * rprof.max())
    dn = cv2.distanceTransform((Mb == 0).astype(np.uint8), cv2.DIST_L2, 5)
    aoz = (keep == 0) & (Mb == 0) & (hgt >= a.hcut) & (dn < 6 * tau)
    if a.no_base_shadow: aoz &= (hgt < yref - 4)
    f_ao = np.where(aoz, 1 - a.ao * np.exp(-dn / tau), 1.0).astype(np.float32)
    # never darker than the darkest nearby original shadow tone would allow
    base = base * f_ao[..., None]
k = A * (hgt >= a.hcut)
k = k * seam * (1 - cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.2))
out = base * (1 - k[..., None]) + Wl * k[..., None]
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(a.out)
if '--dbg' in sys.argv or True:
    np.savez_compressed(a.out.replace('.png', '_masks.npz'), sl=sl, oldM=oldM, Mb=Mb, keep=keep, gain=gain.astype(np.float16), hgt=hgt.astype(np.float32), A=A.astype(np.float16))

dbg = S.astype(np.uint8).copy()
c1, _ = cv2.findContours(Mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(dbg, c1, -1, (255, 0, 0), 2)
yy, xx = np.where(Mb > 0); pad = 60
crop = lambda im: im[max(0, yy.min() - pad):yy.max() + pad, max(0, xx.min() - pad):xx.max() + pad]
Image.fromarray(np.hstack([crop(dbg), crop(np.clip(out, 0, 255).astype(np.uint8))])).save(a.out.replace('.png', '_dbg.jpg'), quality=90)
json.dump(dict(alpha=p[0], s=p[1], tx=p[2], ty=p[3], beta=p[4], taper=p[5], cx0=cx0, yref=yref), open(a.out.replace('.png', '_pose.json'), 'w'), indent=1)
print('done', a.out)
