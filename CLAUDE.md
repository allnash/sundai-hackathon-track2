# CLAUDE.md

## Project Overview

W4A4 (INT4) quantized GEMM CUDA kernel optimization challenge for FLUX.1-schnell diffusion model layers. The goal is to maximize average GEMM TOPs across 4 target shapes.

## Goal: Reach 600 TOPs average GEMM score

## Target GEMM Shapes (M x N x K)

| Layer | M | N | K |
|---|---|---|---|
| attn_to_qkv | 4096 | 9216 | 3072 |
| attn_to_out | 4096 | 3072 | 3072 |
| ff_up | 4096 | 12288 | 3072 |
| ff_down | 4096 | 3072 | 12288 |

## Editable Files

- `your_solution/kernel.cu` — CUDA kernels (quantize + GEMM). **This is what gets timed.**
- `your_solution/quantize.py` — Offline weight quantization (not timed)

**Do NOT modify**: `reference/`, `benchmark.py`, `benchmark.sh`

## Fixed API Signatures

```cpp
std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size);
torch::Tensor gemm_int4_custom(torch::Tensor A_packed, torch::Tensor B_packed,
                                torch::Tensor scales_A, torch::Tensor scales_B, int group_size);
```

## Correctness Constraints

Must pass cosine similarity thresholds vs FP16 matmul:
- attn_to_qkv: > 0.989
- attn_to_out: > 0.991
- ff_up: > 0.978
- ff_down: > 0.977

## Quantization Format

- Packed: `uint8 [rows, K/2]` — two signed INT4 per byte (low nibble = even, high nibble = odd)
- Scales: `float16 [rows, K/group_size]` — one scale per group
- Symmetric per-group: `scale = max(|x|) / 7`, round-to-nearest, clamp [-8, 7]
- Default group_size = 64

## Current Kernel Architecture (v3)

- BLOCK_M=128, BLOCK_N=64, BLOCK_K=128
- 8 warps (256 threads), __launch_bounds__(256, 3) → 3-4 blocks/SM
- 2 quantization groups per K-tile (halves sync count vs BK=64)
- Non-predicated cp.async.cg (L2-only cache, no bounds checks)
- Direct global scale reads with L1 caching (no scale smem)
- Precomputed sa*sb scale products
- Vectorized __half2 epilogue stores
- torch::empty output (no memset)
- ~24.6 KB shared memory per block

## SSH to Benchmark Server

```bash
ssh ubuntu@instance-2044914970637897728.yottadeos.com
```

SSH key is at `~/Downloads/HackathonParticipants`

```bash
ssh -i ~/Downloads/HackathonParticipants ubuntu@instance-2044914970637897728.yottadeos.com
```

## How to Run Benchmark (on server)

```bash
# 1. Activate the conda environment
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate cuda-challenge

# 2. Navigate to project directory (find the right path on the server)
cd /path/to/sundai-hackathon-track2

# 3. Download data (first time only)
python download_data.py

# 4. Run the benchmark
python benchmark.py
```

## Full Workflow: Edit locally → Push → Pull on server → Benchmark

```bash
# LOCAL: After editing kernel.cu, commit and push
git add your_solution/kernel.cu && git commit -m "optimize kernel" && git push

# SERVER: Pull and run
ssh -i ~/Downloads/HackathonParticipants ubuntu@instance-2044914970637897728.yottadeos.com
cd /path/to/sundai-hackathon-track2
git pull
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate cuda-challenge
python benchmark.py
```

## Commit Policy

Only commit when benchmark score goes UP. Include the score in the commit message:
```
git commit -m "Optimize kernel: score XXX TOPs (was YYY)"
```

## External Libraries Allowed

cuBLAS, CUTLASS, Thrust, CUB — all fair game.

## Optimization Roadmap

### Implemented
- [x] MMA tensor core instructions (m16n8k64)
- [x] cp.async double-buffered shared memory
- [x] BLOCK_K=128 (2 groups per tile, halved sync count)
- [x] BLOCK_N=64 for higher occupancy (3-4 blocks/SM)
- [x] Non-predicated cp.async (no branch overhead)
- [x] cp.async.cg (L2-only caching, no L1 pollution)
- [x] Precomputed scale products
- [x] Vectorized __half2 epilogue stores
- [x] torch::empty output
- [x] __launch_bounds__(256, 3) for register control

### Next Optimizations
1. **Warp specialization** — producer/consumer warp split for true compute/memory overlap
2. **3-stage pipeline** — deeper prefetch to hide memory latency
3. **Register-level fragment double buffering** — pre-load next fragments during MMA
4. **Swizzled smem layout** — eliminate 2-way bank conflicts on fragment loads
5. **SplitK for large-K shapes** — ff_down (K=12288) benefits from K-parallel decomposition
6. **CTA rasterization** — Hilbert/swizzle block scheduling for L2 locality
7. **CUTLASS integration** — use CUTLASS 3.x int4 templates as backend
