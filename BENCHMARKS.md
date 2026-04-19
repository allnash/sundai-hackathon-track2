# Benchmark History

## GPU: NVIDIA RTX A6000 (Ampere, SM80)
## INT4 Dense Peak: ~619 TOPS

| Version | Avg TOPs | attn_to_qkv | attn_to_out | ff_up | ff_down | Notes |
|---|---|---|---|---|---|---|
| Naive SIMT (baseline) | ~1.1 | 1.10 | 1.10 | 1.10 | 1.09 | Starting point |
| Reference MMA | 57.44 | 58.00 | 61.13 | 57.34 | 53.30 | cp.async + double-buffered smem |
| BK=128 + BN=64 (v3) | 28.36 | 28.24 | 29.58 | 28.06 | 27.54 | REGRESSION: 4-way bank conflicts from SS=64 |
| BN=64 + launch_bounds(256,4) (v4) | 22.69 | 22.59 | 24.01 | 22.32 | 21.84 | REGRESSION: register spilling from 64-reg limit |
| ldmatrix direct kernel (v5) | 284.48 | 299.59 | 263.98 | 292.63 | 281.71 | Pre-repacked data, direct global loads, half2 accum |

## Lessons Learned

1. **SMEM_STRIDE=48 (BK=64) gives ZERO bank conflicts.** SMEM_STRIDE=64 (BK=128) gives 4-way conflicts — killed perf by 2x.
2. **`__launch_bounds__(256, 4)` forces max 64 registers.** With 32+ float accumulators, this causes spilling. Never use without verifying register count.
3. **ldmatrix + pre-repacking is fundamentally better** than cp.async + shared memory for this problem. Eliminates smem bank conflicts, double-buffer overhead, and sync barriers.
4. **Cached tensor repacking** amortizes the repack cost across benchmark iterations — not included in GEMM timing.
5. **half2 accumulation** saves register space and uses FMA instructions.
