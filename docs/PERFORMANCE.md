# Performance Analysis

## Benchmark Results

GPU: NVIDIA RTX A6000 (Ampere SM86, 84 SMs, 768 GB/s, 619 INT4 TOPS peak)

### Score History

| Version | Avg TOPs | attn_to_qkv | attn_to_out | ff_up | ff_down | Notes |
|---|---|---|---|---|---|---|
| Naive SIMT | ~1.1 | 1.10 | 1.10 | 1.10 | 1.09 | 1 thread per output element |
| Reference MMA | 57.4 | 58.0 | 61.1 | 57.3 | 53.3 | cp.async + smem baseline |
| BK=128 attempt | 28.4 | — | — | — | — | REGRESSION: 4-way bank conflicts |
| BN=64 attempt | 22.7 | — | — | — | — | REGRESSION: register spilling |
| ldmatrix direct v1 | 283.0 | 293.0 | 263.1 | 294.1 | 281.7 | Our first big win |
| + double-buf scales v2 | 297.0 | — | — | — | — | Keep |
| + optimal clipping | 299.0 | — | — | — | — | quantize.py improvement |
| + B-group loop + shfl v3 | 315.0 | 322 | 282 | 318 | 337 | Per-layer group sizes |
| + 4w for attn_to_out | **323.3** | **325** | **309** | **321** | **337** | **Current best** |

### Failed Experiments (see FAILED_EXPERIMENTS.md)

| Experiment | TOPs | Issue |
|---|---|---|
| K-loop unroll ×2 | 227 | Register spilling from code duplication |
| B-fragment prefetch | 297 | No change (compiler already optimizes) |
| v4 agent (swizzle+splitK) | 31 | 3 untested changes at once |
| BM=128 with 8 warps | 239 | Worse MMA-to-overhead ratio |
| Zero-sync (L1 scales) | 102 | Too many global loads |
| __launch_bounds__(256,2) | FAIL | Correctness failure from spilling |
| Hoist B-scale reads | 269 | +16 registers → spilling |

### Per-Layer Analysis (v3, 323 TOPs)

| Layer | M×N×K | TOPs | % Peak | Kernel | Syncs |
|---|---|---|---|---|---|
| attn_to_qkv | 4096×9216×3072 | 325 | 52.5% | 8-warp | 7 |
| attn_to_out | 4096×3072×3072 | 309 | 49.9% | 4-warp | 13 |
| ff_up | 4096×12288×3072 | 321 | 51.9% | 8-warp | 5 |
| ff_down | 4096×3072×12288 | 337 | 54.4% | 8-warp | 9 |

### Why ff_down is Fastest

ff_down (K=12288) has weight group_size=1536, so B-scales only reload 8 times across 192 K-tiles. Each K-tile is nearly sync-free. The high K value also means more compute per byte of B data.

### Why attn_to_out is Slowest

attn_to_out (N=K=3072) has the smallest output (4096×3072) → fewest blocks → worst SM utilization. The 4-warp kernel helps by creating more blocks (32 vs 16 along M-dimension), improving load balancing.

## Key Optimizations and Their Impact

| Optimization | TOPs Gain | Mechanism |
|---|---|---|
| ldmatrix repack | +226 | 1 load vs 4-6, no smem for data |
| half2 FMA accumulation | included | 2× throughput, half the registers |
| Double-buffered scales | +14 | 1 sync/tile instead of 2 |
| A-fragment prefetching | included | Overlap loads with sync barrier |
| Per-layer weight group sizes | +16 | 4-24× fewer __syncthreads |
| __shfl_sync for A scales | included | No smem needed for A scales |
| 4-warp kernel (attn_to_out) | +6 | Better load balancing for small grids |
| Optimal clipping quantize.py | +2 | Better accuracy with same format |

## Register Budget

The #1 constraint. Current: ~125 registers per thread.

```
Accumulators:  AT×NT×4 = 2×8×4 = 64 half2    = 64 registers
A-fragments:   af0, af1 = 2 × uint4           =  8 registers
B-fragment:    bf = 1 × uint4                  =  4 registers
MMA results:   p0,p1,q0,q1 = 4 × int[4]       = 16 registers
Scale vars:    sa0-3, sb01,sb23, s00-s13       = ~15 registers
Addressing:    base, stride, kt, nt, etc.      = ~18 registers
                                          Total ≈ 125 registers
```

At 125 registers: `65536 / (256 × 125) = 2.05` → **2 blocks/SM** = 16 warps.
At 128 registers: drops to 1 block/SM = 8 warps. **Any register increase is catastrophic.**

## Bottleneck Analysis

At 323 TOPs (52% of 619 peak), the remaining 48% gap comes from:

1. **Instruction overhead** (~30%): 27 non-MMA instructions per 4 MMAs per N-tile
   - 8× hmul2 (scale products)
   - 8× __floats2half2_rn (int→half2 conversion)
   - 8× hfma2 (accumulation)
   - 1× global load (B fragment)
   - 2× smem load (B scales)

2. **Memory latency** (~10%): B fragment loads from global memory (L2 hits ~30 cycles)

3. **Synchronization** (~5%): __syncthreads at B-group boundaries

4. **Warp scheduling** (~5%): Only 16 warps per SM (limited by registers)
