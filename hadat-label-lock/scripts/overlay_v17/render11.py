import cv2,numpy as np,sys
exec(open('layers10.py').read().replace("cv2.imwrite(S+'p_txt_sh.png',to_u8(1-txt_sh)); cv2.imwrite(S+'p_txt_co.png',to_u8(1-txt_co))","").replace("gapV=82","gapV=58"))
scene=cv2.imread(S+'clean_scene11.png')[...,::-1].astype(np.float32)   # RGB, cleaned
orig=cv2.imread(S+'../shot1_open_v8_raw.png')[...,::-1].astype(np.float32)
H,W=scene.shape[:2]
def blur(a,s): return cv2.GaussianBlur(a,(0,0),s)

def warp_flat(layer,k,u0,v0,s0,y0,cx,R,box,SS=3):
    bx0,by0,bx1,by1=box
    ls=cv2.resize(layer,None,fx=k*SS,fy=k*SS,interpolation=cv2.INTER_AREA)
    xs=(np.arange((bx1-bx0)*SS)+0.5)/SS+bx0; ys=(np.arange((by1-by0)*SS)+0.5)/SS+by0
    sn=np.clip((xs-cx)/R,-0.9995,0.9995); s=R*np.arcsin(sn)
    u=((s-s0)+u0*k)*SS-0.5; v=((ys-y0)+v0*k)*SS-0.5
    mx=np.tile(u[None,:],(len(ys),1)).astype(np.float32); my=np.tile(v[:,None],(1,len(xs))).astype(np.float32)
    o=cv2.remap(ls,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    inside=(np.abs((xs-cx)/R)<0.9995).astype(np.float32)
    o=o*(inside[None,:,None] if o.ndim==3 else inside[None,:])
    return cv2.resize(o,(bx1-bx0,by1-by0),interpolation=cv2.INTER_AREA)

def ramp(t,stops):
    t=np.clip(t,0,1); ts=np.array([s[0] for s in stops]); out=np.zeros(t.shape+(3,),np.float32)
    for c in range(3): out[...,c]=np.interp(t,ts,[s[1][c] for s in stops])
    return out
GOLD=[(0,(66,44,20)),(0.22,(118,84,38)),(0.46,(178,138,72)),(0.68,(218,184,118)),(0.86,(242,222,172)),(1,(255,247,224))]
SIL_DARK=[(0,(112,102,88)),(0.5,(176,164,144)),(1,(232,222,204))]
SIL_LIGHT=[(0,(88,89,95)),(0.5,(122,123,129)),(1,(170,171,177))]

def gray_alpha(M):
    return np.clip((255-lum(M))/135.0,0,1)*(1-np.clip((sat(M)-12)/25,0,1))

def diamond_layers_cond_sh():
    M=M_co; h,w=lum(M).shape
    Gc=M[...,1]-(M[...,0]+M[...,2])/2
    F=np.clip(Gc/112.0,0,1); Pk=np.clip((M[...,0]-M[...,1])/225.0,0,1)*(M[...,1]<140)
    gr=gray_alpha(M)
    cut=np.zeros((h,w),np.float32); cut[:,:598]=1
    box=np.zeros((h,w),np.float32); box[302:505,253:458]=1
    F*=cut;Pk*=cut
    reg=np.zeros((h,w),np.float32); reg[60:760,110:598]=1
    nearc=cv2.dilate(((F>0.08)|(Pk>0.08)).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
    gr=np.where(gr<0.18,0,gr)
    ink=gr*cut*box; sil=gr*reg*(1-box)*(1-nearc)
    # foil tone from shampoo macket gold (scaled to cond geometry)
    Ms=cv2.resize(M_sh,(int(round(2053*1.0151)),int(round(796*1.0151))),interpolation=cv2.INTER_CUBIC)
    Mt=np.zeros((h,w,3),np.float32); hh=min(h,Ms.shape[0]+1); 
    Mm=cv2.warpAffine(Ms,np.float32([[1,0,0.68],[0,1,-1.28]]),(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    return F,Pk,sil,ink,Mm
F_c,P_c,S_c,Ink_c,Msh_al=diamond_layers_cond_sh()

def fix_tri(S,F,P=None):
    el=lambda n: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
    Sb=(S>0.3).astype(np.uint8)
    nl,lab,st,_=cv2.connectedComponentsWithStats(Sb)
    keep=np.zeros(S.shape,np.float32)
    for i in range(1,nl):
        if st[i,cv2.CC_STAT_AREA]>150: keep[lab==i]=1
    Sd=cv2.dilate(keep,el(5)); Fb=(F>0.15).astype(np.float32); near=cv2.dilate(Fb,el(5))
    gap=Sd*near*(1-keep)*(1-Fb)
    S2=np.maximum(S*keep,gap*0.97)
    S2=np.clip(cv2.GaussianBlur(S2,(0,0),0.6)*1.08,0,1)
    if P is not None:
        P=cv2.GaussianBlur(P,(0,0),0.8)
    return S2,P
S_c,P_c=fix_tri(S_c,F_c,P_c)
# tone map: gold luminance in sh macket where cond foil exists
def tone_map(F,Mgold):
    l=lum(Mgold); wgt=(F>0.5).astype(np.float32)
    # exclude cream/silver triangles in sh: require warm saturation
    warm=((Mgold[...,0]-Mgold[...,2])>18).astype(np.float32); wgt=wgt*warm
    num=blur(l*wgt,7); den=blur(wgt,7)
    t=num/np.maximum(den,1e-3)
    # fill holes with larger blur
    num2=blur(l*wgt,30); den2=blur(wgt,30); t2=num2/np.maximum(den2,1e-3)
    t=np.where(den>0.02,t,t2)
    return np.clip((t-40)/(235-40),0,1)
Tb_c=tone_map(F_c,Msh_al)
# silver/foil for shampoo reuse same geometry
# ----- mask diamond layers -----
def mask_layers():
    M=M_ma; h,w=lum(M).shape
    warm=M[...,0]-M[...,2]
    dia=np.zeros((h,w),np.float32); dia[85:490,130:555]=1
    F=np.clip((warm-6)/24.0,0,1)*dia
    gr=gray_alpha(M)
    box=np.zeros((h,w),np.float32); box[203:386,264:422]=1
    gr=np.where(gr<0.18,0,gr)
    nearf=cv2.dilate((F>0.08).astype(np.uint8),np.ones((5,5),np.uint8)).astype(np.float32)
    ink_logo=gr*dia*box
    sil=gr*dia*(1-box)*(1-nearf)
    txt=gr*(1-dia)
    # drop frame region from text: only inner panel
    inner=np.zeros((h,w),np.float32); inner[26:768,26:655]=1; txt*=inner
    wgt=(F>0.5).astype(np.float32)
    t=blur(lum(M)*wgt,6)/np.maximum(blur(wgt,6),1e-3); t2=blur(lum(M)*wgt,30)/np.maximum(blur(wgt,30),1e-3)
    t=np.where(blur(wgt,6)>0.02,t,t2)
    tb=np.clip((t-50)/(235-50),0,1)
    return F,sil,ink_logo,txt,tb
F_m,S_m,InkL_m,Txt_m,Tb_m=mask_layers()
S_m,_=fix_tri(S_m,F_m)
ink_rgb_m=ink_color(M_ma,gray_alpha(M_ma),np.ones(lum(M_ma).shape,bool),'l')
print('mask ink',ink_rgb_m)
np.savez_compressed(S+'layers_macket.npz',F_c=F_c,P_c=P_c,S_c=S_c,Ink_c=Ink_c,Tb_c=Tb_c,F_m=F_m,S_m=S_m,InkL_m=InkL_m,Txt_m=Txt_m,Tb_m=Tb_m,txt_sh=txt_sh,txt_co=txt_co)
for n,a in [('F_c',F_c),('P_c',P_c),('S_c',S_c),('Ink_c',Ink_c),('F_m',F_m),('S_m',S_m),('InkL_m',InkL_m),('Txt_m',Txt_m)]:
    cv2.imwrite(S+'p_'+n+'.png',to_u8(1-a))
cv2.imwrite(S+'p_Tb_c.png',to_u8(Tb_c))
