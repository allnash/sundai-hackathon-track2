# Performance Analysis

## Benchmark Results

GPU: NVIDIA RTX A6000 (Ampere SM80, 84 SMs, 768 GB/s, 619 INT4 TOPS peak)

### Score History

| Version | Avg TOPs | attn_to_qkv | attn_to_out | ff_up | ff_down | Notes |
|---|---|---|---|---|---|---|
| Naive SIMT | ~1.1 | 1.10 | 1.10 | 1.10 | 1.09 | 1 thread per output element |
| Reference MMA | 57.4 | 58.0 | 61.1 | 57.3 | 53.3 | cp.async + smem baseline |
| BK=128 attempt | 28.4 | — | — | — | — | REGRESSION: 4-way bank conflicts |
| BN=64 attempt | 22.7 | — | — | — | — | REGRESSION: register spilling |
| **ldmatrix direct** | **283.0** | **293.0** | **263.1** | **294.1** | **281.7** | **Current best** |

### Per-Layer Analysis

| Layer | M×N×K | TOPs | % of Peak | Compute (TOPS) | Bytes | AI (ops/byte) |
|---|---|---|---|---|---|---|
| attn_to_qkv | 4096×9216×3072 | 293.0 | 47.3% | 231.9T | ~21GB | 11.0 |
| attn_to_out | 4096×3072×3072 | 263.1 | 42.4% | 77.3T | ~12GB | 6.4 |
| ff_up | 4096×12288×3072 | 294.1 | 47.4% | 309.2T | ~26GB | 11.9 |
| ff_down | 4096×3072×12288 | 281.7 | 45.4% | 309.2T | ~43GB | 7.2 |

All shapes are **compute-bound** (AI >> machine balance of ~0.8 ops/byte).

`attn_to_out` is slowest because it has the smallest output tile (4096×3072 = fewer blocks to fill SMs) and lowest arithmetic intensity.

### Lessons Learned

1. **SMEM_STRIDE matters enormously**. Stride 48 (BK=64) = zero bank conflicts. Stride 64 (BK=128) = 4-way conflicts = 2x slowdown.

2. **`__launch_bounds__` can hurt**. `(256, 4)` forces max 64 registers. With 32+ float accumulators, this causes spilling to local memory = catastrophic slowdown.

3. **ldmatrix eliminates the smem bottleneck entirely**. No cp.async stalls, no bank conflicts, no double buffering overhead. The key insight is that the layout transformation can be cached.

4. **half2 accumulation saves registers and instruction count**. 32 half2 regs vs 64 float regs = more room for the compiler to optimize.

## Roofline Analysis

```
                     Compute Roof (619 TOPS)
                    ─────────────────────────────────────
                   /
                  /
    TOPs        /    ★ Our kernel (283 TOPS, 46%)
               /     
              /      
             /       ─── Memory Roof (768 GB/s)
            /       /
           /       /
          /       /
         ────────
              AI (ops/byte)
```

We are at **46% compute utilization**. The gap to peak comes from:
- K-loop overhead (__syncthreads, scale loads, indexing)
- Global memory latency for fragment loads (not fully hidden)
- FP16 conversion overhead (INT32 → half2 each iteration)
- Warp scheduling inefficiency

## Path to Higher Performance

### Near-term (target: 350-400 TOPs)
- Prefetch fragments into registers before scale sync
- Unroll K-loop by 2 (process 2 K-tiles per sync)
- Use `__launch_bounds__` carefully to hint occupancy without spilling

### Medium-term (target: 400-500 TOPs)
- Warp specialization: 2 producer warps (load data) + 6 consumer warps (MMA)
- 3-4 stage software pipeline for global loads
- StreamK decomposition for better SM utilization

### Theoretical ceiling
- A6000 practical limit for INT4 GEMM: ~520-560 TOPs
- Requires persistent kernel + warp specialization + StreamK
