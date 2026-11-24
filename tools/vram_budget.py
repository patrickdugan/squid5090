"""Runtime-arena VRAM estimate, mirroring src/engine.zig compute_runtime_vram_bytes.

Usage: python tools/vram_budget.py [--free-gib 15.36] [--weights-gib 13.70]
Prints runtime GiB and total for each (prefill batch, kv format, ctx) combination and
the largest ctx that fits in the given free VRAM. Pure arithmetic; touches no GPU.
"""
import argparse

A = 256
H, KV, Q, HEADS, KVH, GR, E, TOPK, INTER = 2304, 1024, 3072, 24, 8, 64, 128, 8, 1152
VOCAB = 200058
QRGS = Q + 16 * HEADS + Q + HEADS
SWA, SUB, RING, MTPB, DSWS, SPLIT = 1024, 256, 16, 8, 2_000_000, 2_050_000
MAX_VERIFY_B = 8
EXPERT_GROUPS = 4 + MAX_VERIFY_B * TOPK * (4 + 1 + MAX_VERIFY_B + 4 * MAX_VERIFY_B)


def al(x):
    return (x + A - 1) // A * A


def kv_pair(cap, fmt):
    if fmt == "bf16":
        return 2 * al(cap * KV * 2)
    if fmt == "fp8":
        return 2 * al(cap * KV) + 2 * al(cap * KVH * 2)
    return 2 * al(cap * KV // 2) + 2 * al(cap * 16 * 2)


def runtime_bytes(ctx, kv, pb, dspark=True):
    t = 0
    for layer_visit in range(48):
        phys = layer_visit if layer_visit < 6 else (6 + (layer_visit - 6) % 12 if layer_visit < 42 else layer_visit - 24)
        glob = (phys + 1) % 6 == 0
        t += kv_pair(ctx if glob else SWA + SUB, kv if glob else "bf16")
        t += 2 * al(RING * KV * 4) + 2 * al(RING * H * 4)
    if dspark:
        for ml in range(3):
            loc = ml in (0, 2)
            cap = SWA + MTPB if loc else (65536 if ctx > 524288 else ctx)
            t += kv_pair(cap, "bf16" if loc else kv)
        t += al(RING * H * 4)
    t += al(SPLIT * 4)
    b = MTPB
    t += al(4) * 4 + al(b * 4) * 3 + al(7 * 4) * 2 + al(4)
    t += 3 * al(b * H * 4) + al(b * 4 * H * 4) + 2 * al(b * H * 4) + 3 * (al(b * H * 4) + al(b * H * 2))
    t += al(b * QRGS * 4) + 2 * al(b * KV * 4) + al(b * Q * 4) + al(b * Q * 2) + 4 * al(b * H * 4) + al(b * H * 2)
    t += al(3 * b * H * 4) + al(3 * b * 2 * H * 4) + al(3 * b * GR * 4) + al(b * E * 4) + 2 * al(b * TOPK * 4)
    t += al(EXPERT_GROUPS) + al(b * TOPK * INTER * 2) + al(b * H * 4) + al(b * TOPK * H * 4) + al(4 * b * H * 4)
    t += al(3 * b * H * 4) + al(b * 2 * KV * 4) + al(DSWS * 4) + 2 * al(b * 1024 * 4) + al(b * VOCAB * 4) + al(32)
    t += al(pb * 4) + al(pb // 8 * 4) + 3 * al(pb * H * 4) + al(pb * 4 * H * 4) + 2 * al(pb * H * 4)
    t += 3 * (al(pb * H * 4) + al(pb * H * 2)) + al(pb * QRGS * 4) + 2 * al(pb * KV * 4) + al(pb * Q * 2)
    t += 4 * al(pb * H * 4) + al(pb * H * 2) + al(pb * E * 4) + 2 * al(pb * TOPK * 4) + al(1024 * 4)
    t += 3 * al(pb * TOPK * 4) + al(pb * TOPK * INTER * 2) + al(pb * TOPK * H * 4) + al(4 * pb * H * 4)
    t += al(QRGS * H * 2) + al(pb * H * 4) + al(pb * 2 * H * 2) + al(pb * GR * 4) + al(pb * GR * 2)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--free-gib", type=float, default=15.36, help="NVML free VRAM inside the process (idle 5080 Laptop under WSL2: 15.36)")
    ap.add_argument("--weights-gib", type=float, default=13.70, help="nvfp4 weights with host embed (LEARNINGS.md 2.2)")
    ap.add_argument("--reserve-gib", type=float, default=0.375, help="SQUID_VRAM_RESERVE_MIB default 384")
    args = ap.parse_args()
    gib = 1024 ** 3
    budget = (args.free_gib - args.weights_gib - args.reserve_gib) * gib
    print(f"runtime budget after weights + reserve: {budget / gib:.2f} GiB")
    print(f"{'prefill':>7} {'kv':>4} {'ctx':>8} {'runtime GiB':>11} {'total GiB':>9}")
    for pb in (2048, 1024, 512):
        for kv in ("fp8", "q4"):
            for ctx in (8192, 32768, 65536):
                r = runtime_bytes(ctx, kv, pb) / gib
                print(f"{pb:>7} {kv:>4} {ctx:>8} {r:>11.2f} {r + args.weights_gib:>9.2f}")
            lo, hi = 4, 1024
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if runtime_bytes(mid * 1024, kv, pb) <= budget:
                    lo = mid
                else:
                    hi = mid - 1
            fit = lo * 1024 if runtime_bytes(lo * 1024, kv, pb) <= budget else 0
            print(f"{pb:>7} {kv:>4} max ctx that fits: {fit}")


if __name__ == "__main__":
    main()
