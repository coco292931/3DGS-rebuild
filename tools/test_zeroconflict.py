"""验证：空槽位全部指向索引 0 造成的极端原子冲突，是不是 backward 的真凶。"""
import time
import torch

dev = "cuda"
P, K, N = 76800, 20, 30000


def bench(fn, n=8, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def run(gidx, label):
    src = torch.rand(N, 3, device=dev)
    def f():
        s = src.clone().requires_grad_(True)
        # 模拟真实链路：多次 gather 同一个索引
        a = s[gidx]
        b = s[gidx].flatten()
        c = s[gidx].sum(dim=1)
        (a.sum() + b.sum() + c.sum()).backward()
    t = bench(f)
    print(f"{label:<38} {t:8.2f} ms")
    return t


g_uniform = torch.randint(0, N, (P, K), device=dev)

g_half_zero = torch.randint(0, N, (P, K), device=dev)
mask = torch.rand(P, K, device=dev) < 0.49
g_half_zero[mask] = 0

# 分散版：把「空槽位」指向各不相同的合法索引（梯度本就是 0，指哪都安全）
g_scattered = g_half_zero.clone()
idx = torch.arange(P * K, device=dev).reshape(P, K) % N
g_scattered[mask] = idx[mask]

t1 = run(g_uniform, "全部均匀随机")
t2 = run(g_half_zero, "49% 指向索引 0（当前实现）")
t3 = run(g_scattered, "49% 分散到不同索引（改法）")
print(f"\n>>> 集中比均匀慢 {t2/t1:.1f}x ；分散后比集中快 {t2/t3:.1f}x")
