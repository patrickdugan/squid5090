# squid5090

**A single-binary, zero-Python TigerStyle Zig + Blackwell CUDA (`sm_120a`) inference engine and OpenAI-compatible API server for [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) — hitting `589.5 tok/s` code decode (`771.6 tok/s` on code-copy/RAG), `9,181 tok/s` prefill, and `1,048,576` (1M) context on a single RTX 5090.**

![squid5090 RTX 5090 Benchmark & 1M Context Scaling](assets/benchmark.png)

> **Running on a 16 GiB RTX 5080?** See **[`PORT-5080.md`](PORT-5080.md)** (CUDA 12.8+ builds, WSL2, VRAM auto-fit).

> **Want to understand how it works?**
> - **[`LITE.md`](LITE.md)** — Plain-English, 3Blue1Brown-style visual guide to every optimization and how runtime-gated abliteration works step by step.
> - **[`LEARNINGS.md`](LEARNINGS.md)** — Complete technical reference: every Blackwell kernel optimization, roofline audit vs. [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill) (including what's still left on the table), and the `uv` abliteration pipeline.

---

## Why `squid5090` on an RTX 5090?

Single-user (`c=1`) comparison for **Inkling-2-Mini** on one **NVIDIA RTX 5090 (32 GiB GDDR7)**:

| Inference Engine | Runtime Stack | Decode Speed (`c=1`) | Max Context (32 GiB) | VRAM / Host Spill | Key Benefits & Tradeoffs |
|---|---|---:|---:|---|---|
| **PyTorch / HF** | Python + PyTorch | ~10–15 tok/s | ~64K | >24 GiB | Reference baseline; eager Python overhead |
| **`llama.cpp` / Ollama** | C++ (GGUF) | ~90–97 tok/s | ~256K | ~18–22 GiB | No native NVFP4 Blackwell PTX or DSpark MTP |
| **vLLM / SGLang** | Python + CUDA | ~190–350 tok/s | ~256K–512K | ~22–28 GiB | Built for multi-user batching; Python overhead at `c=1` |
| **Basalt** *(same 5090)* | Native NVFP4 + Q8 | 261–325 tok/s | 256K | 22.9 GiB + **8.9 GiB RAM spill** | Spills weights to host RAM; capped at 256K |
| **`ingine`** *(published ref)* | C++ + CUDA | 561 tok/s *(chat: 325)* | 256K | ~18–22 GiB | Adaptive `{2,4,8}` DSpark; capped at 256K |
| **`squid5090` (this repo)** | **1 Zig binary (`0` Python)** | **589.5 tok/s** code<br>**771.6 tok/s** code-copy/RAG<br>**332.5 tok/s** chat<br>**541.1 tok/s** ablated | **1,048,576 (1M)** | **15.6 GiB** (8K)<br>**23.2 GiB** (1M)<br>**0 GiB RAM spill** | **2 `cudaMalloc` arenas**, zero post-init `malloc`, **9,181 tok/s** prefill (`8,755` at 20K, `7,742` at 64K), **62–72 ms** warm prefix cache, optional 1.8 MB `--ablation` |

---

## Quickstart (Linux Build & Install)

### 1. Linux Requirements

- **OS:** Linux x86_64 (tested on Ubuntu 24.04 / Debian 12–13)
- **GPU:** NVIDIA Blackwell GPU (`sm_120a`, e.g., RTX 5090) with ≥ 16 GiB free VRAM (≥ 23.5 GiB for full 1M context)
- **NVIDIA Driver:** `580+` (tested on `590.48.01`)
- **CUDA Toolkit:** CUDA `13.0+` with `nvcc` and `cuBLAS` (tested on CUDA `13.1` at `/usr/local/cuda`)
- **Build Tools:** `g++`, `git`, `curl`, and **Zig `0.17.1+`**

```bash
# Install host C++ compiler (required by nvcc) and basics on Debian/Ubuntu
sudo apt-get update && sudo apt-get install -y build-essential git curl

# Verify CUDA 13+ and Zig 0.17.1+ are on PATH
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
zig version
```

*Runtime links only against `libc`, `libstdc++`, `libcudart`, `libcublas`, and `libpthread` — zero Python, CMake, Make, or external packages.*

### 2. Clone & Build

```bash
git clone git@github.com:tensorfish/squid5090.git
cd squid5090

# Compiles all sm_120a CUDA kernels in parallel (content-hashed cache) + ReleaseFast Zig
zig build -Doptimize=ReleaseFast
```

The single binary is built at `zig-out/bin/squid5090` (~30 s clean build, ~24 ms incremental).

### 3. Download Inkling-2-Mini Weights

Download [`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (~20.2 GB) into a local folder (e.g., `/opt/models/inkling-2-mini`):

```bash
uvx --from huggingface-hub hf download thinkingmachines/Inkling-2-Mini \
  --local-dir /opt/models/inkling-2-mini
```

### 4. Run the OpenAI API Server or CLI

```bash
# Full 1M-context OpenAI HTTP/SSE server (~23.2 GiB VRAM, adaptive {2,4,8} DSpark + prompt-lookup)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 --spec-threshold 0.80

# Optional: enable 1.8 MB runtime-gated abliteration (zero weight modification)
./zig-out/bin/squid5090 serve /opt/models/inkling-2-mini \
  --host 0.0.0.0 --port 8000 --mode nvfp4 --kv q4 --ctx 1048576 --mtp 7 \
  --ablation /opt/models/inkling-2-mini/inkling-2-mini-abliterated.safetensors

# One-shot CLI generation
./zig-out/bin/squid5090 generate /opt/models/inkling-2-mini \
  --prompt "Write a Python LRU cache with tests." \
  --mode nvfp4 --kv q4 --mtp 7 --max-tokens 1024

# Benchmark & 1M long-context scaling sweep
./zig-out/bin/squid5090 bench /opt/models/inkling-2-mini \
  --prompt "Write a Python LRU cache with tests." --mode nvfp4 --kv q4 --mtp 7 --max-tokens 1024 --runs 3
./zig-out/bin/squid5090 longctx /opt/models/inkling-2-mini --mode nvfp4 --kv q4 --ctx 1048576
```

**Test the server (`curl` or OpenAI Python SDK):**
```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Write a 1-line Python function to reverse a string."}],
    "max_tokens": 256,
    "stream": true
  }'
```

**Install as a persistent `systemd` service:**
```bash
sudo cp deploy/squid5090.service /etc/systemd/system/squid5090.service
sudo systemctl daemon-reload && sudo systemctl enable --now squid5090.service
curl -s http://localhost:8000/health
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--mode` | `nvfp4` (`serve`) | Weight precision: `nvfp4` \| `fp8` \| `exact` |
| `--kv` | `auto` | Global KV cache format: `bf16` \| `fp8` (up to 432K) \| `q4` (up to **1M**) |
| `--ctx` | `65536` (`serve`) | Context window size up to `1048576` |
| `--mtp` | `7` (`serve`) | Built-in DSpark drafter (`0` = off, `7` = adaptive `{2,4,8}` CUDA Graph switch) |
| `--spec-threshold` | `0.80` | Markov confidence threshold for `{2,4,8}` verify width (`0.0` = fixed `B=8`) |
| `--no-prompt-lookup` | off | Disable hybrid O(1) prompt-lookup drafter (enabled by default) |
| `--ablation` | none | Optional 1.8 MB runtime-gated Fisher-LDA `.safetensors` (see [`tools/ablate.py`](tools/ablate.py)) |
| `--api-key` | `$SQUID_API_KEY` | Optional `Authorization: Bearer <key>` auth for `serve` |

---

## Documentation & Attribution

- **[`LITE.md`](LITE.md)** — High-level, 3Blue1Brown-style plain-English guide to LLM optimization and abliteration.
- **[`LEARNINGS.md`](LEARNINGS.md)** — Consolidated technical reference: GPU kernels, roofline audit vs. [`sybil-solutions/inference-optimisation-skill`](https://github.com/sybil-solutions/inference-optimisation-skill), unexploited future levers, and the `uv` abliteration pipeline.
- **[`NOTICE`](NOTICE)** — Open-source attributions (`satellitedown/inkling-2-mini-abliterated-ingine`, `AtomicBot-ai/atomic-engine`, `sybil-solutions/inference-optimisation-skill`, `thinkingmachines/Inkling-2-Mini`).
