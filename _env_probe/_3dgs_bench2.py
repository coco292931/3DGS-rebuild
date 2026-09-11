import torch, time, json
torch.manual_seed(0)
dev='cuda'; W,H=400,300
def run(N, log_s_lo, log_s_hi, K=12, tag=""):
    means=torch.randn(N,3,device=dev)*1.5; means[:,2]=means[:,2].abs()+2.0
    q=torch.randn(N,4,device=dev); q=q/q.norm(dim=1,keepdim=True)
    log_s=torch.rand(N,3,device=dev)*(log_s_hi-log_s_lo)+log_s_lo
    opac=torch.rand(N,1,device=dev); col=torch.rand(N,3,device=dev)
    for p in (means,q,log_s,opac,col): p.requires_grad_(True)
    fx=fy=400.0; cx,cy=W/2,H/2
    viewmat=torch.eye(4,device=dev); viewmat[2,3]=5.0
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0=time.time()
    w,x,y,z=q.unbind(1)
    r0=torch.stack([1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],1)
    r1=torch.stack([2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],1)
    r2=torch.stack([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],1)
    R=torch.stack([r0,r1,r2],1); M=R*torch.exp(log_s).unsqueeze(1); Sig=M@M.transpose(1,2)
    t=torch.cat([means,torch.ones(N,1,device=dev)],1)@viewmat.t()
    tx,ty,tz=t[:,0],t[:,1],torch.clamp(t[:,2],min=1e-4)
    T=(torch.stack([torch.stack([fx/tz,torch.zeros_like(tz),-fx*tx/tz**2],1),
                    torch.stack([torch.zeros_like(tz),fy/tz,-fy*ty/tz**2],1),
                    torch.zeros(N,3,device=dev)],1))@viewmat[:3,:3]
    cov=T@Sig@T.transpose(1,2); cov[:,0,0]+=0.3; cov[:,1,1]+=0.3
    det=torch.clamp(cov[:,0,0]*cov[:,1,1]-cov[:,0,1]*cov[:,0,1],min=1e-9)
    inv=torch.stack([cov[:,1,1]/det,-cov[:,0,1]/det,-cov[:,1,0]/det,cov[:,0,0]/det],1)
    mid=(cov[:,0,0]+cov[:,1,1])/2
    rad=3.0*torch.sqrt(torch.clamp(mid+torch.sqrt(torch.clamp(mid*mid-det,min=0)),min=1e-9))
    torch.cuda.synchronize(); t1=time.time()
    px=fx*tx/tz+cx; py=fy*ty/tz+cy
    x0=torch.clamp((px-rad).floor().long(),0,W-1); x1=torch.clamp((px+rad).ceil().long(),0,W-1)
    y0=torch.clamp((py-rad).floor().long(),0,H-1); y1=torch.clamp((py+rad).ceil().long(),0,H-1)
    nw=torch.where((x1>=x0)&(y1>=y0)&(rad>0.3),(x1-x0+1)*(y1-y0+1),torch.zeros(N,dtype=torch.long,device=dev))
    total=int(nw.sum().item())
    gid=torch.repeat_interleave(torch.arange(N,device=dev),nw,output_size=total)
    starts=torch.cumsum(nw,0)-nw
    off=torch.arange(total,device=dev)-starts[gid]
    wpx=(x1-x0+1)[gid]
    pid=(y0[gid]+off//wpx)*W+(x0[gid]+off%wpx)
    key=pid.to(torch.int64)*(1<<22)+(tz[gid]*4096).to(torch.int64)
    order=torch.argsort(key); gs=gid[order]; ps=pid[order]
    slot=torch.arange(total,device=dev)-torch.searchsorted(ps,ps)
    keep=slot<K
    gsk=gs[keep]; psk=ps[keep]; slotk=slot[keep]
    sparse=torch.zeros(W*H,K,dtype=torch.long,device=dev)
    sparse[psk,slotk]=gsk+1
    valid_mask=sparse>0; gidx=(sparse-1).clamp(min=0)
    cxx=torch.arange(W,device=dev).view(1,W).expand(H,W).reshape(-1)
    cyy=torch.arange(H,device=dev).view(H,1).expand(H,W).reshape(-1)
    dx=cxx.unsqueeze(1)-px[gidx]; dy=cyy.unsqueeze(1)-py[gidx]
    iv=inv[gidx]
    power=-0.5*(iv[...,0]*dx*dx+iv[...,3]*dy*dy)-iv[...,1]*dx*dy
    alpha=torch.sigmoid(opac[gidx][...,0])*torch.exp(torch.clamp(power,max=0.0))*valid_mask
    om=1.0-alpha
    Tp=torch.cumprod(om,1)/torch.clamp(om,min=1e-8)
    wt=alpha*Tp
    img=(wt.unsqueeze(-1)*col[gidx]).sum(1).reshape(H,W,3)
    torch.cuda.synchronize(); t2=time.time()
    img.pow(2).mean().backward()
    torch.cuda.synchronize(); t3=time.time()
    return {"tag":tag,"N":N,"frags":total,"proj_ms":round((t1-t0)*1000,1),"raster_ms":round((t2-t1)*1000,1),
            "bwd_ms":round((t3-t2)*1000,1),"peak_MB":round(torch.cuda.max_memory_allocated()/1e6,1)}
r=[run(100000,-5.6,-4.6,12,"small"), run(150000,-6.0,-5.0,12,"tiny"), run(60000,-5.2,-4.2,12,"medium-ish")]
open(r"C:\Users\tonyp\Downloads\_bench2.json","w").write(json.dumps(r,indent=1))
print(json.dumps(r,indent=1))
