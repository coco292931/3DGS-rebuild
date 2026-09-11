import torch, math, time, json
torch.manual_seed(0)
dev = 'cuda'
W, H, N = 400, 300, 100000

means = torch.randn(N, 3, device=dev) * 1.5
means[:, 2] = means[:, 2].abs() + 2.0
q = torch.randn(N, 4, device=dev); q = q / q.norm(dim=1, keepdim=True)
log_s = (torch.rand(N, 3, device=dev) * 0.4 - 3.2)   # small gaussians
opac = torch.rand(N, 1, device=dev)
col = torch.rand(N, 3, device=dev)
viewmat = torch.eye(4, device=dev)
viewmat[2, 3] = 5.0
fx = fy = 400.0
cx, cy = W / 2, H / 2
BG = torch.zeros(3, device=dev)

def quat_to_R(q):
    w, x, y, z = q.unbind(1)
    r0 = torch.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], 1)
    r1 = torch.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], 1)
    r2 = torch.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1)
    return torch.stack([r0, r1, r2], 1)

def project(means, q, log_s, opac, col):
    R = quat_to_R(q)
    S = torch.exp(log_s)
    M = R * S.unsqueeze(1)
    Sigma = M @ M.transpose(1, 2)
    ones = torch.ones(N, 1, device=dev)
    p_h = torch.cat([means, ones], 1)
    t = p_h @ viewmat.t()
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
    tz = torch.clamp(tz, min=1e-4)
    J = torch.zeros(N, 3, 3, device=dev)
    J[:, 0, 0] = fx / tz; J[:, 1, 1] = fy / tz
    J[:, 0, 2] = -fx * tx / tz**2; J[:, 1, 2] = -fy * ty / tz**2
    Wm = viewmat[:3, :3]
    T = J @ Wm
    cov = T @ Sigma @ T.transpose(1, 2)
    cov[:, 0, 0] += 0.3; cov[:, 1, 1] += 0.3
    det = cov[:, 0, 0]*cov[:, 1, 1] - cov[:, 0, 1]**2
    det = torch.clamp(det, min=1e-9)
    inv = torch.stack([cov[:, 1, 1]/det, -cov[:, 0, 1]/det, -cov[:, 1, 0]/det, cov[:, 0, 0]/det], 1)
    mid = (cov[:, 0, 0] + cov[:, 1, 1]) / 2
    rad = 3.0 * torch.sqrt(torch.clamp(mid + torch.sqrt(torch.clamp(mid**2 - det, min=0)), min=1e-9))
    return tx, ty, tz, inv, rad

def raster(tx, ty, tz, inv, rad, opac, col, K=12):
    px = fx * tx / tz + cx
    py = fy * ty / tz + cy
    x0 = torch.clamp((px - rad).floor().long(), 0, W-1)
    x1 = torch.clamp((px + rad).ceil().long(), 0, W-1)
    y0 = torch.clamp((py - rad).floor().long(), 0, H-1)
    y1 = torch.clamp((py + rad).ceil().long(), 0, H-1)
    valid = (x1 >= x0) & (y1 >= y0) & (tz > 0.01) & (rad > 0.3)
    nw = (x1 - x0 + 1) * (y1 - y0 + 1)
    nw = torch.where(valid, nw, torch.zeros_like(nw))
    total = int(nw.sum().item())
    gid = torch.repeat_interleave(torch.arange(N, device=dev), nw, output_size=total)
    starts = torch.cumsum(nw, 0) - nw
    off = torch.arange(total, device=dev) - starts[gid]
    wpx = (x1 - x0 + 1)[gid]
    fy_ = (y1 - y0 + 1)[gid]
    fyid = (y0)[gid] + off // wpx
    fxid = (x0)[gid] + off % wpx
    pid = fyid * W + fxid
    depth = tz[gid]
    key = pid.to(torch.int64) * (1 << 22) + (depth * 4096).to(torch.int64)
    order = torch.argsort(key)
    gs = gid[order]; ps = pid[order]
    slot = torch.arange(total, device=dev) - torch.searchsorted(ps, ps)
    keep = slot < K
    gsk = gs[keep]; psk = ps[keep]; slotk = slot[keep]
    sparse = torch.zeros(P_TOT, K, dtype=torch.long, device=dev)
    sparse[psk, slotk] = gsk + 1
    valid_mask = sparse > 0
    gidx = (sparse - 1).clamp(min=0)
    gpx = px[gidx]; gpy = py[gidx]
    cxx = torch.arange(W, device=dev).view(1, W).expand(H, W).reshape(-1)
    cyy = torch.arange(H, device=dev).view(H, 1).expand(H, W).reshape(-1)
    dx = cxx.unsqueeze(1) - gpx
    dy = cyy.unsqueeze(1) - gpy
    iv = inv[gidx]
    power = -0.5 * (iv[..., 0]*dx*dx + iv[..., 3]*dy*dy) - iv[..., 1]*dx*dy
    alpha = torch.sigmoid(opac[gidx][..., 0]) * torch.exp(torch.clamp(power, max=0.0))
    alpha = alpha * valid_mask
    one_minus = 1.0 - alpha
    Tp = torch.cumprod(one_minus, 1) / torch.clamp(one_minus, min=1e-8)
    wts = alpha * Tp
    rgb = (wts.unsqueeze(-1) * col[gidx]).sum(1) + Tp[:, -1:].clamp(min=0) * BG
    return rgb.reshape(H, W, 3)
for p in (means, q, log_s, opac, col):
    p.requires_grad_(True)
P_TOT = W * H
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
out = project(means, q, log_s, opac, col)
t1 = time.time()
img = raster(*out, opac, col)
t2 = time.time()
loss = img.pow(2).mean()
loss.backward()
t3 = time.time()
torch.cuda.synchronize()
print(json.dumps({
  "forward_proj_ms": round((t1-t0)*1000, 1),
  "forward_raster_ms": round((t2-t1)*1000, 1),
  "backward_ms": round((t3-t2)*1000, 1),
  "peak_mem_MB": round(torch.cuda.max_memory_allocated()/1e6, 1),
  "img_mean": float(img.mean()),
}))
