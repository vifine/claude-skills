"""label_cyl.py - wrapper to use label_flat.py on a CYLINDRICAL bottle: unwrap the label band of the
scene into a flat strip, fix the label there with label_flat.py, wrap the change back onto the cylinder.
Only pixels changed by label_flat are written back; the rest of the scene stays as generated.
Used first on 112501 "31 washes" (frame 1666:120, 20.10.2026): dark label, bottle upright,
label_rewrap.py gave grey paper and dim print, this route kept the generated label and replaced only the text.

Steps (bottle must be upright; if tilted, rotate the scene first - same convention as label_rewrap.py):
  1) python3 label_cyl.py unwrap scene.png flat.png --edges XL,XR --rows Y0,Y1 [--theta 1.3] [--pad 20]
       flat.png: label band unwrapped (x = arc length), padded by --pad px left/right.
       Prints the label_flat quad for given label top/bottom rows (--label T,B).
  2) python3 label_cyl.py artcrop label.png artc.png --edges XL,XR --k K --uc U [--theta 1.3]
       crops the visible part of the flat label export (transparent -> black) to the unwrapped width.
       K = label height in scene / label export height; U = export x facing the camera.
  3) python3 label_flat.py flat.png artc.png flatfix.png --quad "<printed quad>" --only "<boxes in artc px>" --dark ...
  4) python3 label_cyl.py rewrap scene.png flat.png flatfix.png out.png --edges XL,XR --rows Y0,Y1
       [--theta 1.3] [--pad 20] [--clean x0,y0,x1,y1]
       --clean: zone in flat px (without pad) where light residue of erased old print is pulled down
       to the local paper level (old text that was larger / placed differently than the new text).

Example 112501 (scene 2048, upright):
  unwrap  scene.png flat.png --edges 1101,1436 --rows 900,1500 --label 1010,1411
  artcrop labA.png artc.png --edges 1101,1436 --k 0.1683 --uc 1600
  label_flat.py flat.png artc.png flatfix.png --quad "20,110;456,110;456,511;20,511" \
      --only "475,910,1020,1465;1440,200,2588,1500" --dark --strip 10 --light 1.5
  rewrap  scene.png flat.png flatfix.png out.png --edges 1101,1436 --rows 900,1500 --clean 235,150,436,300
"""
import argparse, numpy as np, cv2
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['unwrap', 'artcrop', 'rewrap'])
ap.add_argument('files', nargs='+')
ap.add_argument('--edges', required=True, help='body left,right x at the label rows')
ap.add_argument('--rows', default=None, help='scene rows Y0,Y1 of the band to unwrap (a bit above / below the label)')
ap.add_argument('--theta', type=float, default=1.3, help='max angle (rad) unwrapped; edges beyond stay from the scene')
ap.add_argument('--pad', type=int, default=20, help='px padding left/right of the flat strip (label_flat needs room around the quad)')
ap.add_argument('--label', default=None, help='unwrap: label top,bottom rows in the scene -> prints the label_flat quad')
ap.add_argument('--k', type=float, default=None); ap.add_argument('--uc', type=float, default=None)
ap.add_argument('--clean', default=None)
a = ap.parse_args()
XL, XR = map(float, a.edges.split(','))
cx, R = (XL + XR) / 2, (XR - XL) / 2
smax = R * a.theta; Wf = int(2 * smax) + 1
rd = lambda p: np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
sv = lambda x, p: Image.fromarray(np.clip(x, 0, 255).astype(np.uint8)).save(p)

if a.mode == 'unwrap':
    scene, flat = a.files; Y0, Y1 = map(int, a.rows.split(','))
    S = rd(scene)
    xf = np.arange(Wf, dtype=np.float32) - smax
    MX = np.repeat((cx + R * np.sin(xf / R)).astype(np.float32)[None], Y1 - Y0, 0)
    MY = np.repeat(np.arange(Y0, Y1, dtype=np.float32)[:, None], Wf, 1)
    F = cv2.remap(S, MX, MY, cv2.INTER_CUBIC)
    sv(np.pad(F, ((0, 0), (a.pad, a.pad), (0, 0)), mode='edge'), flat)
    print('flat', Wf + 2 * a.pad, 'x', Y1 - Y0)
    if a.label:
        T, B = map(float, a.label.split(','))
        t, b = T - Y0, B - Y0; l, r = a.pad, a.pad + Wf
        print('quad "%d,%d;%d,%d;%d,%d;%d,%d"' % (l, t, r, t, r, b, l, b))

elif a.mode == 'artcrop':
    src, dst = a.files
    im = Image.open(src).convert('RGBA'); bg = Image.new('RGBA', im.size, (0, 0, 0, 255)); bg.alpha_composite(im)
    u0, u1 = a.uc - smax / a.k, a.uc + smax / a.k
    bg.convert('RGB').crop((int(round(u0)), 0, int(round(u1)), im.size[1])).save(dst)
    print('art crop u', round(u0), '..', round(u1))

else:
    scene, flat0, flatfix, out = a.files; Y0, Y1 = map(int, a.rows.split(','))
    S = rd(scene); F0 = rd(flat0)[:, a.pad:-a.pad]; F = rd(flatfix)[:, a.pad:-a.pad]
    if a.clean:
        x0, y0, x1, y1 = map(int, a.clean.split(','))
        F8 = np.clip(F, 0, 255).astype(np.uint8)
        Bk = cv2.GaussianBlur(cv2.morphologyEx(F8, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(np.float32), (0, 0), 2)
        Pp = cv2.GaussianBlur(cv2.erode(F8, np.ones((21, 21), np.uint8)).astype(np.float32), (0, 0), 8) + 4
        Z = np.zeros(F.shape[:2], np.float32); Z[y0:y1, x0:x1] = 1; Z = cv2.GaussianBlur(Z, (0, 0), 5)[..., None]
        F = F - np.clip(Bk - Pp, 0, None) * Z
    H, W = S.shape[:2]; xs = np.arange(W, dtype=np.float32)
    xf = R * np.arcsin(np.clip((xs - cx) / R, -0.9999, 0.9999)) + smax
    MX = np.repeat(xf[None], Y1 - Y0, 0).astype(np.float32)
    MY = np.repeat(np.arange(Y1 - Y0, dtype=np.float32)[:, None], W, 1)
    D = cv2.remap(F - F0, MX, MY, cv2.INTER_CUBIC)
    ok = ((xs > XL + 1) & (xs < XR - 1) & (np.abs(xs - cx) / R < np.sin(a.theta) - 0.01)).astype(np.float32)
    S[Y0:Y1] += D * ok[None, :, None]
    sv(S, out); print('done', out)
