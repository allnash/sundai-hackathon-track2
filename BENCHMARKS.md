# Benchmark History

## GPU: NVIDIA RTX A6000 (Ampere SM86, 84 SMs, 619 INT4 TOPS peak)

| Version | Avg TOPs | attn_to_qkv | attn_to_out | ff_up | ff_down | Notes |
|---|---|---|---|---|---|---|
| Naive SIMT | ~1.1 | 1.10 | 1.10 | 1.10 | 1.09 | Starting point |
| Reference MMA | 57.4 | 58.0 | 61.1 | 57.3 | 53.3 | cp.async + double-buffered smem |
| BK=128 + SS=64 | 28.4 | 28.2 | 29.6 | 28.1 | 27.5 | REGRESSION: 4-way bank conflicts |
| BN=64 + launch_bounds | 22.7 | 22.6 | 24.0 | 22.3 | 21.8 | REGRESSION: register spilling |
| ldmatrix direct v1 | 283.0 | 293.0 | 263.1 | 294.1 | 281.7 | Pre-repacked fragments, half2 FMA |
| + double-buf scales v2 | 297.0 | — | — | — | — | 1 sync/tile, A-frag prefetch |
| + optimal clipping | 299.0 | — | — | — | — | MSE-optimal quantize.py |
| + B-group + shfl v3 | 315.0 | 322 | 282 | 318 | 337 | Per-layer group sizes |
| **+ 4w attn_to_out** | **323.3** | **325** | **309** | **321** | **337** | **Current best** |

## Current: 323 TOPs (52% of peak)

See `docs/` for full architecture documentation.
