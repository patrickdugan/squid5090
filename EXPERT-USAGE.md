# Expert usage on the RTX 5080 (Inkling-2-Mini, nvfp4)

Measured 2026-11-25 to decide whether offloading MoE experts to host RAM would pay on a 16 GiB card.

## How it was measured

`SQUID_EXPERT_TRACE=<path>` (off by default) appends every routing decision to a zero-copy pinned
host buffer: one 16-byte record per routed token per layer visit (visit, request, phase, token in
batch, top-8 expert ids), written in stream order by `cuda/expert_trace.cu` after the router and
dumped at engine shutdown. The request index is read from device memory, so the CUDA-graph decode
path records it correctly. `tools/expert_usage.py` analyses the dump and simulates a per-layer
expert cache in VRAM.

Run: `serve --mode nvfp4` (MTP 7, ctx 65536 / kv q4), greedy, 512 max tokens, eight requests:
code (Python LRU cache), prose (short story), math (word problem), translation (es/fr/zh), chat
(burnout advice), JSON (fictional employees), a 16,344-token source dump to summarize, and code
(Rust Dijkstra). 1,060,560 records: 794,496 prefill token-visits and 266,064 decode token-visits
(3,810 generated tokens in 1,664 forward passes, 3.33 tokens per verify pass, 2.29 accepted).

Each expert is 4.27 MiB in nvfp4 (w13 + w2 with scales); the 24 x 128 experts are 12.8 GiB of the
13.69 GiB of weights.

## Results

| | prefill | decode |
|---|---:|---:|
| routings covered by each layer's top 16 / 32 / 64 experts | 47% / 68% / 89% | 34% / 53% / 78% |
| effective experts per layer (exp entropy, of 128) | 69 (min 37, layer 18) | 96 (min 64, layer 17) |
| experts never routed to | 5 of 3,072 | 0 of 3,072 |

Decode routing is broad, and the hot set moves with the task: each request's top-32 experts per
layer overlap the other seven requests' pooled top-32 with a mean Jaccard of 0.22 (Rust code) to
0.32 (translation). A verify pass touches 16.3 distinct experts per layer visit, 3.27 GiB of expert
weights per forward pass.

Simulated per-layer expert cache over the decode stream (misses are expert loads over PCIe):

| experts kept per layer | VRAM freed | LRU loads / pass | LRU PCIe / pass | fixed hot set loads / pass (in-sample) |
|---:|---:|---:|---:|---:|
| 112 | 1.60 GiB | 6.7 | 29 MiB | 20.5 |
| 96 | 3.20 GiB | 17.4 | 75 MiB | 63.4 |
| 80 | 4.80 GiB | 37.3 | 159 MiB | 122.6 |
| 64 | 6.41 GiB | 71.1 | 304 MiB | 197.0 |
| 48 | 8.01 GiB | 130.4 | 557 MiB | 291.0 |
| 32 | 9.61 GiB | 264.3 | 1,129 MiB | 405.5 |

A fixed hot set chosen from the same trace does worse than LRU at every size, so an offload scheme
would need a dynamic cache, not a pinned hot list.

## What it means

Decode here runs at about 15-18 ms per forward pass. The GPU's link reports PCIe 4.0 x16 max and
x8 current; at the ~13 GB/s an x8 link moves in practice (not measured here), LRU traffic per pass
would add about 2 ms at 112 experts per layer, 6 ms at 96 and 23 ms at 64, before any overlap of
copies with compute. Keeping 112 or 96 experts per layer frees 1.6 or 3.2 GiB for KV (two to three
times the current 1.21 GiB runtime arena) at an estimated 10-35% decode cost; going lower costs
more than the whole current pass time.
