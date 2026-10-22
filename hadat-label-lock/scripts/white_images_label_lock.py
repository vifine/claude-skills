"""HADAT label lock v6 (current).
Hero0 (ref) stays pixel-exact; the AI scene is warped INTO the Hero0 frame.
Bottle + label = Hero0 pixels above band_top; below band_top = AI (glossy reflections).
Foreground objects (nuts, petals) = AI, detected as components that cross the bottle contour.
Background: illumination flattened to pure #FFFFFF, contact shadows and pale petals kept.
usage: python3 -I lock6.py hero0_rgb.png bottle_alpha.png ai.png out.png band_top occ_y
  band_top: y where AI reflections start (just below the label); occ_y: y below which foreground objects are searched
  (set occ_y below the label bottom for light labels so the label is never replaced)"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage
rp,ap,aip,op=sys.argv[1:5]; bt=int(sys.argv[5]) if len(sys.argv)>5 else 1655; OCC_Y=int(sys.argv[6]) if len(sys.argv)>6 else 1400
R=np.array(Image.open(rp).convert('RGB')).astype(np.float32); H,W=R.shape[:2]
alpha=np.array(Image.open(ap).convert('RGBA').resize((W,H)))[...,3]>128
alpha=ndimage.binary_erosion(alpha,iterations=3)
A=np.array(Image.open(aip).convert('RGB').resize((W,H),Image.LANCZOS)).astype(np.float32)
g=lambda x: cv2.GaussianBlur(cv2.cvtColor(x.astype(np.uint8),cv2.COLOR_RGB2GRAY),(0,0),1.5).astype(np.float32)/255
yy=np.arange(H)[:,None]
mk=((alpha&(yy<bt))*255).astype(np.uint8); mk=cv2.dilate(mk,np.ones((21,21),np.uint8))
warp=np.eye(3,dtype=np.float32); gr,ga=g(R),g(A)
for s in (0.25,0.5,1.0):
    sr=cv2.resize(gr,None,fx=s,fy=s); sa=cv2.resize(ga,None,fx=s,fy=s); m=cv2.resize(mk,None,fx=s,fy=s)
    Ws=warp.copy(); Ws[0,2]*=s; Ws[1,2]*=s; Ws[2,0]/=s; Ws[2,1]/=s
    _,Ws=cv2.findTransformECC(sr,sa,Ws,cv2.MOTION_HOMOGRAPHY,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,200,1e-6),m,5)
    warp=Ws.copy(); warp[0,2]/=s; warp[1,2]/=s; warp[2,0]*=s; warp[2,1]*=s
Aw=cv2.warpPerspective(A,warp,(W,H),flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
lum=lambda x: cv2.cvtColor(np.clip(x,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)
sat=lambda x: x.max(-1)-x.min(-1)
LA,SA=lum(Aw),sat(Aw)
# occluders: brown nut / creamy butter pixels in AI inside the bottle silhouette, lower part
diff=np.abs(Aw-R).max(-1)
LM=lum(R)
cand=(yy>OCC_Y)&(((diff>35)&(SA>20))|(LA>150))
cand=ndimage.binary_opening(cand,iterations=2)
lab,n=ndimage.label(cand)
inside=ndimage.sum(alpha,lab,range(1,n+1)); total=ndimage.sum(cand,lab,range(1,n+1))
keep_ids=[k+1 for k in range(n) if inside[k]>300 and (total[k]-inside[k])>400]
obj=np.isin(lab,keep_ids)
obj=ndimage.binary_fill_holes(ndimage.binary_closing(obj,iterations=4))
occ=alpha&obj
# reference pixels: bottle above band, minus occluders
keep=(alpha&(yy<bt)&~occ).astype(np.float32)
# soft transition into the AI lower band
ramp=np.clip((bt-yy)/60.0,0,1)*np.ones((1,W)); keepF=np.minimum(cv2.GaussianBlur(keep,(0,0),1.2), np.where(yy<bt-60,1,ramp))*(~occ)
keepF=cv2.GaussianBlur(keepF.astype(np.float32),(0,0),1.0)
out=Aw*(1-keepF[...,None])+R*keepF[...,None]
# background: gentle lift so paper white becomes #FFFFFF; pale objects (petals) keep their tones
LO=lum(out)
cand=(LO>215)&(out.max(-1)-out.min(-1)<10)
cand=cand&~cv2.dilate(((alpha|occ)).astype(np.uint8),np.ones((41,41),np.uint8)).astype(bool)
cf=cand.astype(np.float32)
field=np.stack([cv2.GaussianBlur(np.where(cand,out[...,c],0).astype(np.float32),(0,0),60)/np.maximum(cv2.GaussianBlur(cf,(0,0),60),1e-3) for c in range(3)],-1)
field=np.clip(field,200,255)
bgL=float(np.median(field))
lifted=np.clip(out*255.0/field,0,255)
keepb=cv2.GaussianBlur((alpha&~occ).astype(np.float32),(0,0),1.0)[...,None]
out=out*keepb+lifted*(1-keepb)
out=np.where((out.min(-1,keepdims=True)>=250)&(keepb<0.5),255.0,out)
out=np.clip(out,0,255).astype(np.uint8)
Image.fromarray(out).save(op)
print('warp',np.round(warp,4).tolist(),'bgL',bgL,'occ px',int(occ.sum()),'white share',(out.min(-1)==255).mean())
