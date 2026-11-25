"""Analyse a SQUID_EXPERT_TRACE dump: expert-usage skew and what an expert cache in VRAM would cost.

Record layout (cuda/expert_trace.cu): int32 visit, int32 tag (request << 20 | phase << 16 | token
in batch), uint8 ids[8]. Phase 1 = batched prefill; phase 0 = decode steps (and short prompts that
prefill through the batch-8 graph). Usage: python tools/expert_usage.py trace.bin [labels.json]
"""

import json
import sys
from collections import OrderedDict

import numpy as np

EXPERTS = 128
TOP_K = 8
LAYERS = 24
LOOP_START, LOOP_END, LOOP_PASSES = 6, 18, 3
# nvfp4 expert bank per expert: w13 [2*1152, 2304] + w2 [2304, 1152], 4-bit weights + one fp8 scale per 16.
EXPERT_BYTES = (2 * 1152 * 2304 + 2304 * 1152) * 9 // 16


def visit_to_layer():
    out = list(range(LOOP_START))
    for _ in range(LOOP_PASSES):
        out += list(range(LOOP_START, LOOP_END))
    out += list(range(LOOP_END, LAYERS))
    assert len(out) == 48
    return np.array(out)


def load(path):
    rec = np.fromfile(path, dtype=np.dtype([("visit", "<i4"), ("tag", "<i4"), ("ids", "u1", (8,))]))
    req = rec["tag"] >> 20
    phase = (rec["tag"] >> 16) & 0xF
    tok = rec["tag"] & 0xFFFF
    return rec["visit"], req, phase, tok, rec["ids"].astype(np.int64)


def steps(visit, tok):
    """Group consecutive records into kernel launches (one routing call = one step at one visit)."""
    starts = np.flatnonzero((tok == 0) | np.r_[True, visit[1:] != visit[:-1]])
    return np.r_[starts, len(visit)]


def simulate(visit, ids, bounds, layer_of, cache_slots, policy, hot=None):
    """Per-physical-layer expert cache: count expert loads (misses) over the decode stream."""
    caches = [OrderedDict() for _ in range(LAYERS)]
    misses = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        layer = layer_of[visit[a]]
        need = np.unique(ids[a:b])
        if policy == "static":
            misses += int(np.sum(~np.isin(need, hot[layer])))
            continue
        c = caches[layer]
        for e in need.tolist():
            if e in c:
                c.move_to_end(e)
            else:
                misses += 1
                c[e] = True
                if len(c) > cache_slots:
                    c.popitem(last=False)
    return misses


def main():
    path = sys.argv[1]
    labels = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    visit, req, phase, tok, ids = load(path)
    layer_of = visit_to_layer()
    layer = layer_of[visit]
    print(f"records {len(visit):,}  requests {sorted(set(req.tolist()))}  expert {EXPERT_BYTES / 2**20:.2f} MiB")

    for name, sel in (("prefill", phase == 1), ("decode", phase == 0)):
        if not sel.any():
            continue
        counts = np.zeros((LAYERS, EXPERTS))
        np.add.at(counts, (np.repeat(layer[sel], TOP_K), ids[sel].ravel()), 1)
        share = np.sort(counts, axis=1)[:, ::-1].cumsum(axis=1) / counts.sum(axis=1, keepdims=True)
        p = counts / counts.sum(axis=1, keepdims=True)
        eff = np.exp(-(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1))
        unused = (counts == 0).sum(axis=1)
        print(f"\n== {name}: {sel.sum():,} token-visits")
        print("  routings covered by the top 16/32/64 experts of each layer (mean over layers): "
              + " / ".join(f"{share[:, k - 1].mean():.1%}" for k in (16, 32, 64)))
        print(f"  effective experts per layer (exp entropy): mean {eff.mean():.1f} of 128, min {eff.min():.1f} (layer {eff.argmin()}), max {eff.max():.1f}")
        print(f"  experts never used: {unused.sum()} of {LAYERS * EXPERTS} (max {unused.max()} in one layer)")
        print("  per layer top-32 share: " + " ".join(f"{s:.0%}" for s in share[:, 31]))

    # Domain stability: overlap of each request's top-32 decode experts per layer with the pooled set.
    dec = phase == 0
    if labels:
        print("\n== per-request decode hot sets (top-32 per layer): mean Jaccard vs. the other requests pooled")
        reqs = sorted(set(req[dec].tolist()))
        tops = {}
        for r in reqs:
            s = dec & (req == r)
            c = np.zeros((LAYERS, EXPERTS))
            np.add.at(c, (np.repeat(layer[s], TOP_K), ids[s].ravel()), 1)
            tops[r] = c
        for r in reqs:
            rest = sum(tops[o] for o in reqs if o != r)
            j = []
            for l in range(LAYERS):
                a = set(np.argsort(-tops[r][l])[:32].tolist())
                b = set(np.argsort(-rest[l])[:32].tolist())
                j.append(len(a & b) / len(a | b))
            lab = labels[r - 1] if 0 < r <= len(labels) else str(r)
            print(f"  req {r} ({lab}): {np.mean(j):.2f}")

    # Cache simulation on the decode stream.
    v, i, t = visit[dec], ids[dec], tok[dec]
    bounds = steps(v, t)
    n_steps = len(bounds) - 1
    batch = np.diff(bounds)
    # Token positions advanced: one per verify step at layer visit 0 is a lower bound; use generated
    # tokens from the requests instead when the caller has them. Report per decode step (all 48 visits).
    per_token_steps = n_steps / 48
    union = np.array([len(np.unique(i[a:b])) for a, b in zip(bounds[:-1], bounds[1:])])
    print(f"\n== decode stream: {per_token_steps:,.0f} forward passes, mean batch {batch.mean():.2f} tokens, "
          f"mean distinct experts per layer visit {union.mean():.1f} (of {TOP_K} x batch)")
    counts = np.zeros((LAYERS, EXPERTS))
    np.add.at(counts, (np.repeat(layer_of[v], TOP_K), i.ravel()), 1)
    full_bytes = union.sum() * EXPERT_BYTES
    print(f"  expert bytes read per forward pass with everything resident: {full_bytes / per_token_steps / 2**30:.2f} GiB")
    print("  slots/layer  VRAM for experts   LRU misses/pass  static-hot misses/pass  PCIe per pass (LRU)")
    for slots in (32, 48, 64, 80, 96, 112):
        hot = [np.argsort(-counts[l])[:slots] for l in range(LAYERS)]
        lru = simulate(v, i, bounds, layer_of, slots, "lru")
        st = simulate(v, i, bounds, layer_of, slots, "static", hot)
        vram = slots * LAYERS * EXPERT_BYTES / 2**30
        print(f"  {slots:>10}  {vram:>12.2f} GiB  {lru / per_token_steps:>15.1f}  {st / per_token_steps:>22.1f}  "
              f"{lru * EXPERT_BYTES / per_token_steps / 2**20:>12.0f} MiB")


if __name__ == "__main__":
    main()
