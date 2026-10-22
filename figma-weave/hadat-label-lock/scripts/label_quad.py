"""Paste Hero0 label (rect 700,898-1300,1645) onto a quad in a scene; lighting transfer; skin/occluder protection.
usage: scene.png hero.png out.png "[[TLx,TLy],[TRx,TRy],[BRx,BRy],[BLx,BLy]]" [transfer]"""
import sys,numpy as np,cv2
from PIL import Image
from scipy import ndimage
sp,hp,op=sys.argv[1:4]; pts=np.float32(eval(sys.argv[4])); TR=float(sys.argv[5]) if len(sys.argv)>5 else 0.5
S=np.array(Image.open(sp).convert('RGB')).astype(np.float32); H=np.array(Image.open(hp).convert('RGB')).astype(np.float32)
h,w=S.shape[:2]; src=np.float32([[700,898],[1300,898],[1300,1645],[700,1645]])
Hm=cv2.getPerspectiveTransform(src,pts); W=cv2.warpPerspective(H,Hm,(w,h),flags=cv2.INTER_AREA)
m=np.zeros(H.shape[:2],np.float32); m[898:1645,700:1300]=1; M=cv2.warpPerspective(m,Hm,(w,h)); Mb=(M>0.5).astype(np.float32)
lm=lambda a: cv2.cvtColor(np.clip(a,0,255).astype(np.uint8),cv2.COLOR_RGB2GRAY).astype(np.float32)+8
def blr(x,s):
    wgt=Mb if x.ndim==2 else Mb[...,None]; d=cv2.GaussianBlur(Mb,(0,0),s)+1e-3
    return cv2.GaussianBlur(x*wgt,(0,0),s)/(d if x.ndim==2 else d[...,None])
with np.errstate(all='ignore'):
    ratio=np.nan_to_num(np.clip(blr(lm(S),20)/blr(lm(W),20),0.4,2.0),nan=1.0)[...,None]
    tint=np.nan_to_num(np.clip((blr(S,25)+6)/(blr(W,25)+6)/ratio,0.8,1.3),nan=1.0)
Wl=np.clip(W*(1+(ratio*tint-1)*TR),0,255)
L=lm(S)-8; skin=(L>70)&(S[...,0]>S[...,2]+25)
lab,n=ndimage.label(skin); outside=ndimage.binary_dilation(Mb<0.5,iterations=2)
ids=[i for i in np.unique(lab[outside&skin]) if i>0]; occ=np.isin(lab,ids)
if len(sys.argv)>6:
    yy,xx=np.mgrid[0:h,0:w]; occ&=eval(sys.argv[6])
occ=cv2.dilate(occ.astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)
k=cv2.erode(Mb,np.ones((3,3),np.uint8))*(1-occ); k=cv2.GaussianBlur(k,(0,0),1.0)
out=S*(1-k[...,None])+Wl*k[...,None]; Image.fromarray(out.astype(np.uint8)).save(op)
x0,y0=pts.min(0).astype(int)-60; x1,y1=pts.max(0).astype(int)+60
a=Image.fromarray(S.astype(np.uint8)).crop((x0,y0,x1,y1)); b=Image.open(op).crop((x0,y0,x1,y1))
c=Image.new('RGB',(a.width*2+10,a.height));c.paste(a,(0,0));c.paste(b,(a.width+10,0)); c=c.resize((c.width*2,c.height*2)); c.save('cmp.jpg',quality=92)
