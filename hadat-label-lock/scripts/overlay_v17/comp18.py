exec(open('/tmp/claude-0/-home-claude/9e97d4eb-0303-5f86-a445-536615652a81/scratchpad/lab/render11.py').read().replace('clean_scene11.png','base_clean.png'))

# ===== geometric triangles (vector polygons, 4x supersampled) =====
def fit_tri(mask):
    cnt,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    c=max(cnt,key=cv2.contourArea); pts=c.reshape(-1,2).astype(np.float64)
    ap=cv2.approxPolyDP(c,3.0,True).reshape(-1,2).astype(np.float64); assert len(ap)==3
    lines=[]
    for i in range(3):
        a,b=ap[i],ap[(i+1)%3]; d=b-a; L=np.linalg.norm(d); d/=L; n=np.array([-d[1],d[0]])
        t=(pts-a)@d; dist=np.abs((pts-a)@n)
        sel=(t>0.15*L)&(t<0.85*L)&(dist<4)
        vx,vy,x0,y0=cv2.fitLine(pts[sel].astype(np.float32),cv2.DIST_L2,0,0.01,0.01).ravel()
        lines.append((np.array([x0,y0],np.float64),np.array([vx,vy],np.float64)))
    return lines
def inter(l1,l2):
    p1,d1=l1;p2,d2=l2; A=np.array([d1,-d2]).T; t=np.linalg.solve(A,p2-p1); return p1+t[0]*d1
def offset_lines(lines,cen,off):
    out=[]
    for p,d in lines:
        n=np.array([-d[1],d[0]])
        if (p-cen)@n<0: n=-n
        out.append((p+n*off,d))
    return out
def poly_from(lines): return np.array([inter(lines[i],lines[(i+1)%3]) for i in range(3)])
def raster_poly(polys,shape,SSF=4):
    h,w=shape; im=np.zeros((h*SSF,w*SSF),np.uint8)
    for P in polys:
        q=((P+0.5)*SSF-0.5)*16     # pixel-centre convention
        cv2.fillPoly(im,[np.round(q).astype(np.int32)],255,lineType=cv2.LINE_8,shift=4)
    return cv2.resize(im.astype(np.float32)/255,(w,h),interpolation=cv2.INTER_AREA)
def geo_layer(Sold,Fold,under,shape_dbg=None):
    Sb=(Sold>0.5).astype(np.uint8); nl,lab,st,_=cv2.connectedComponentsWithStats(Sb)
    ex=[];of=[]
    for i in range(1,nl):
        if st[i,4]<150: continue
        ls=fit_tri(lab==i); cen=np.argwhere(lab==i)[:,::-1].mean(0)
        ex.append(poly_from(offset_lines(ls,cen,0.5)))
        of.append(poly_from(offset_lines(ls,cen,0.5+under)))
    h,w=Sold.shape
    Pe=raster_poly(ex,(h,w)); Po=raster_poly(of,(h,w))
    Fcl=cv2.dilate((Fold>0.2).astype(np.uint8),np.ones((3,3),np.uint8)).astype(np.float32)
    return np.clip(np.maximum(Pe,Po*Fcl),0,1),ex,Pe
def geo_plain(Pold):
    Pb=(Pold>0.5).astype(np.uint8); nl,lab,st,_=cv2.connectedComponentsWithStats(Pb); ex=[]
    for i in range(1,nl):
        if st[i,4]<60: continue
        ls=fit_tri(lab==i); cen=np.argwhere(lab==i)[:,::-1].mean(0); ex.append(poly_from(offset_lines(ls,cen,0.5)))
    return raster_poly(ex,Pold.shape),ex
S_c_old,S_m_old,P_c_old=S_c.copy(),S_m.copy(),P_c.copy()
UNDER=5.0   # macket px (~2.6 px in frame)
S_c,ex_c,Se_c=geo_layer(S_c_old,F_c,UNDER); S_m,ex_m,Se_m=geo_layer(S_m_old,F_m,UNDER); P_c,ex_p=geo_plain(P_c_old)
print('geo tri c',[p.round(1).tolist() for p in ex_c]); print('geo tri m',[p.round(1).tolist() for p in ex_m]); print('geo lac',[p.round(1).tolist() for p in ex_p])
out=scene.copy()
LAYERS={}   # name -> RGBA float full-frame for export

SIL_ONBLACK=[(0,(112,118,128)),(0.5,(158,165,176)),(1,(198,205,216))]
SIL_LIGHT=[(0,(100,105,112)),(0.5,(142,147,155)),(1,(182,187,195))]
def lighting(X,Y,cxL,RL,box):
    th=np.arcsin(np.clip((X-cxL)/RL,-0.999,0.999))
    lf=0.5+0.5*np.clip(np.cos(th+0.61),0,1)
    spec=np.exp(-((th+0.50)/0.20)**2)
    bx0,by0,bx1,by1=box; Wd=bx1-bx0; Hh=by1-by0
    gl=np.exp(-(((X-(bx0+0.30*Wd))/(0.34*Wd))**2+((Y-(by0+0.22*Hh))/(0.30*Hh))**2))
    return th,lf,spec,gl

def emboss(a,s=0.85):
    ab=blur(a,s); ax=cv2.Sobel(ab,cv2.CV_32F,1,0,ksize=3)/8; ay=cv2.Sobel(ab,cv2.CV_32F,0,1,ksize=3)/8
    d=(ax+ay)*1.5
    return np.clip(d,0,1),np.clip(-d,0,1)


SIZ=(H,W)
def put(name,a,c,x0,y0,x1,y1):
    L=np.zeros((H,W,4),np.float32); L[y0:y1,x0:x1,:3]=c; L[y0:y1,x0:x1,3]=np.clip(a,0,1)*255
    LAYERS[name]=L
def render_label(name,cfg):
    bx0,by0,bx1,by1=cfg['box']; box=cfg['box']
    ys,xs=np.mgrid[by0:by1,bx0:bx1].astype(np.float32); X=xs+0.5; Y=ys+0.5
    th,lf,spec,gl=lighting(X,Y,cfg['cxL'],cfg['RL'],box)
    base=scene[by0:by1,bx0:bx1].copy()
    dark=cfg['dark']
    lx0,lx1,ly0,ly1=cfg['clip']
    clip=((X>lx0)&(X<lx1)&(Y>ly0)&(Y<ly1)).astype(np.float32)
    clip=clip*np.load(S+'inside11.npz')[name][by0:by1,bx0:bx1].astype(np.float32); clip=blur(clip,0.7)
    def wl(layer,grp):
        g=cfg[grp]
        return warp_flat(layer,g['k'],g['u0'],g['v0'],g['s0'],g['y0'],cfg['cx'],cfg['R'],box)*(clip if layer.ndim==2 else clip[...,None])
    d=cfg['dia']
    Fa=np.clip(wl(cfg['F'],'dia'),0,1); Sa=np.clip(wl(cfg['S'],'dia'),0,1)
    Tb=warp_flat(cfg['Tb'],d['k'],d['u0'],d['v0'],d['s0'],d['y0'],cfg['cx'],cfg['R'],box)
    Wf=warp_flat((cfg['F']>0.3).astype(np.float32),d['k'],d['u0'],d['v0'],d['s0'],d['y0'],cfg['cx'],cfg['R'],box)
    Tb=np.clip(np.where(Wf>0.05,Tb/np.maximum(Wf,0.05),0.5),0,1)
    Pa=np.clip(wl(cfg['P'],'dia'),0,1) if cfg.get('P') is not None else np.zeros_like(Fa)
    Ia=np.clip(wl(cfg['InkL'],'dia'),0,1)
    Ta=np.clip(wl(cfg['Txt'],'txt')*cfg.get('tboost',1.0),0,1)
    t=np.clip((X-lx0)/(lx1-lx0),0,1); dd=np.minimum(t,1-t)/0.085
    Eg=np.clip(dd,0,1); Eg=Eg*Eg*(3-2*Eg)
    thp=np.arcsin(np.clip((X-cfg['cxL'])/cfg['RL'],-0.999,0.999)); cosf=np.clip(np.cos(thp),0,1)**0.6
    lm=(Eg*(0.55+0.45*cosf)).astype(np.float32)
    Fa=Fa*lm; Sa=Sa*lm; Pa=Pa*lm; Ia=Ia*lm; Ta=Ta*lm
    rng=np.random.default_rng(7)
    # silver satin
    grain=cv2.GaussianBlur(rng.normal(0,1,Sa.shape).astype(np.float32),(0,0),0.55)*0.03
    diag=np.clip(0.5+0.0030*((X-bx0)-(Y-by0)),0,1)
    ts=np.clip(0.20+0.46*lf+0.22*diag+0.30*gl+0.12*spec+grain,0,1)
    ts=blur(ts,0.0001) if False else ts
    silc=ramp(ts,SIL_ONBLACK if dark else SIL_LIGHT)
    # printed metallic ink (champagne gold), flat: broad satin gradient from the bottle light map, no rim/emboss
    Hh_=by1-by0; Wd_=bx1-bx0
    vert=1-(Y-by0)/Hh_
    tf0=np.clip(0.20+0.42*lf+0.30*gl+0.10*vert+0.10*(Tb-0.5)+0.10*spec,0,1)
    ph=((X-bx0)*0.0105+(Y-by0)*0.0062+0.9*th)
    wave=0.5+0.5*np.sin(2*np.pi*ph)
    tf=np.clip(0.46+(tf0-tf0.mean())*1.8+0.30*(wave-0.5),0,1)
    tf=blur(tf,1.6)
    FOIL=[(0.0,(100,74,42)),(0.22,(124,95,54)),(0.50,(172,146,92)),(0.78,(216,190,136)),(1.0,(238,220,176))]
    foilc=ramp(tf,FOIL)
    foilc=foilc*(0.84+0.16*np.clip(np.cos(th),0,1)**0.5)[...,None]
    fr=np.random.default_rng(11)
    shim=cv2.GaussianBlur(fr.normal(0,1,Fa.shape).astype(np.float32),(0,0),0.6)*0.035
    foilc=np.clip(foilc*(1+shim)[...,None],0,255)
    pg=cv2.GaussianBlur(fr.normal(0,1,Fa.shape).astype(np.float32),(0,0),0.8)
    Fa=Fa*np.clip(0.90+0.045*pg,0.78,0.97)    # ink not fully opaque: label paper grain shows through
    Cf=foilc
    put(name+'_foil',Fa,Cf,bx0,by0,bx1,by1)
    Sew=np.clip(wl(cfg['Se'],'dia'),0,1)
    wmix=np.clip(Sew/np.maximum(Sa,1e-3),0,1)[...,None]
    silc=silc*wmix+Cf*(1-wmix)      # underlap band takes the gold colour: nothing grey shows through the ink grain
    put(name+'_silver',Sa,silc,bx0,by0,bx1,by1)

    if dark:
        # label_stock: black matte paper, strictly the label contour, fading to the cylinder edge
        stk=clip*(0.93*lm/np.maximum(lm.max(),1e-3))
        lfs=0.5+0.5*np.clip(np.cos(th+0.61),0,1)
        paper=(np.array([9,9,10],np.float32)[None,None,:]+ (6*lfs+3*spec+2*gl)[...,None]*np.array([1,0.99,0.96],np.float32))
        put(name+'_stock',stk,np.broadcast_to(paper,base.shape),bx0,by0,bx1,by1)
    inkA=np.clip(Ia+Ta-Ia*Ta,0,1)
    if dark:
        col=cfg['ink']*(0.50+0.46*lf)[...,None]
        put(name+'_inkd',0.93*inkA,col,bx0,by0,bx1,by1)
    else:
        put(name+'_inkl',inkA,np.broadcast_to(cfg['ink'],base.shape),bx0,by0,bx1,by1)
    if cfg.get('P') is not None:
        refl=np.clip(spec*0.95+gl*0.65,0,1)
        tint=np.array([1.0,0.97,0.92]) if dark else np.array([0.86,0.95,1.0])
        Ld=(34*refl if dark else (5+22*refl))[...,None]*tint      # additive light wanted
        Sv=np.clip(Ld*255/np.maximum(255-base,8),0,255)           # Screen colour giving that add
        put(name+'_lacq',Pa,Sv,bx0,by0,bx1,by1)
def text_anchor(canvas,k,yc,s0):
    return dict(k=k,u0=0,v0=0,s0=s0,y0=yc-canvas.shape[0]*k/2)

# ---------- SHAMPOO ----------
ink_sh=np.array([206,196,178],np.float32)
cfg_sh=dict(box=(1183,833,1539,1267),clip=(1192,1530,840,1260),cx=1361.0,R=170.0,cxL=1361.0,RL=170.0,dark=True,
    F=F_c,S=S_c,Se=Se_c,P=P_c,InkL=Ink_c,Tb=Tb_c,Txt=txt_sh,ink=ink_sh)
kd=421/796.0
cfg_sh['dia']=dict(k=kd,u0=349.0,v0=398.0,s0=-85.0,y0=1049.5)
cfg_sh['txt']=dict(k=0.37,u0=0,v0=0,s0=50.0,y0=930.0); cfg_sh['tboost']=1.3
render_label('sh',cfg_sh)
# ---------- CONDITIONER ----------
ink_co=np.array([92,92,98],np.float32)
cfg_co=dict(box=(1639,851,1996,1283),clip=(1647,1988,858,1277),cx=1817.5,R=171.5,cxL=1817.5,RL=171.5,dark=False,
    F=F_c,S=S_c,Se=Se_c,P=P_c,InkL=Ink_c,Tb=Tb_c,Txt=txt_co,ink=ink_co)
cfg_co['dia']=dict(k=420/806.0,u0=354.0,v0=403.0,s0=-85.0,y0=1067.0)
cfg_co['txt']=dict(k=0.37,u0=0,v0=0,s0=50.0,y0=947.0); cfg_co['tboost']=1.25
render_label('co',cfg_co)
# ---------- MASK ----------
cfg_ma=dict(box=(2152,792,2470,1164),clip=(2161,2461,801,1155),cx=2311.0,R=837.0,cxL=2299.0,RL=211.0,dark=False,
    F=F_m,S=S_m,Se=Se_m,P=None,InkL=InkL_m+Txt_m,Tb=Tb_m,Txt=np.zeros_like(Txt_m),ink=np.array([117,119,123],np.float32))
cfg_ma['dia']=dict(k=0.4736,u0=340.5,v0=0.0,s0=0.0,y0=790.0)
cfg_ma['txt']=cfg_ma['dia']
render_label('ma',cfg_ma)


import os
OUT=S+'v17_layers/'; os.makedirs(OUT,exist_ok=True)
def save(nm,arr):
    a=np.clip(arr,0,255).astype(np.uint8)
    import PIL.Image as I; I.fromarray(a,'RGBA').save(OUT+nm+'.png',optimize=True)
def comb(ks):
    o=np.zeros((H,W,4),np.float32)
    for k in ks:
        a=LAYERS[k]; m=a[...,3:]>o[...,3:]
        o=np.where(m,a,o)   # regions are disjoint between bottles
    return o
save('label_stock',comb(['sh_stock']))
save('foil',comb(['sh_foil','co_foil','ma_foil']))
save('silver',comb(['co_silver','ma_silver','sh_silver']))
save('ink_dark',comb(['sh_inkd']))
save('ink_light',comb(['co_inkl','ma_inkl']))
save('lacquer_uv',comb(['sh_lacq','co_lacq']))
print('layers done')
