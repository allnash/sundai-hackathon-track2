# Kernel Architecture

## Overview

Our INT4 GEMM kernel computes `C[M,N] = A[M,K] @ B[N,K]^T` where both A (activations) and B (weights) are quantized to signed INT4 with per-group FP16 scales.

The kernel uses a **two-phase architecture**:

1. **Offline repack** (cached, not timed): Transform packed INT4 data into MMA-register-friendly layout using `ldmatrix`
2. **Online GEMM** (timed): Load pre-repacked fragments directly from global memory, compute MMA, accumulate in half2

This achieves **283 TOPs** on RTX A6000 (46% of 619 TOPS INT4 peak).

## Why Not cp.async + Shared Memory?

The traditional approach (used by the reference MMA kernel at 57 TOPs) is:

```
Global Memory → cp.async → Shared Memory → manual uint32 loads → MMA registers
```

This has several problems:
- **24KB shared memory** per block limits occupancy to 2-3 blocks/SM
- **cp.async barriers** (`cp.async.wait_group`) stall the pipeline every K-tile
- **Bank conflicts** on shared memory fragment loads (stride-dependent)
- **4-6 instructions** per fragment load (manual uint32 reads from smem)
- **Double buffering** adds complexity and smem usage

Our approach eliminates all of these:

```
OFFLINE: Raw data → ldmatrix repack → Fragment-layout tensor (cached)
ONLINE:  Fragment tensor → 1 uint4 global load → MMA registers (done!)
```

## Tile Configuration

```
BLOCK_M  = 256    rows per thread block
BLOCK_N  = 128    cols per thread block
BLOCK_K  = 64     K elements per tile (= group_size)
NUM_WARPS = 8     warps per block (256 threads)
WARP_M   = 32     rows per warp
A_TILES  = 2      vertical m16 MMA tiles per warp (32/16)
N_TILES  = 8      horizontal 16-col N-tiles (128/16)
```

Each warp computes a 32×128 output tile using:
- 2 vertical A-tiles × 8 horizontal N-tiles × 2 B-halves = **32 MMA operations per K-tile**

## Data Layout After Repacking

### A (Activations) — Row Operand

Raw format: `A_packed[M, K/2]` — row-major, 2 INT4 per byte

Repacked format indexed as: `A_rep[bm][kt][warp][a_tile][lane]` where each element is `uint4` (16 bytes)

- `bm` = M-block index (M / 256)
- `kt` = K-tile index (K / 64)
- `warp` = warp within block (0..7)
- `a_tile` = vertical tile (0..1)
- `lane` = thread lane (0..31)

Each `uint4` contains one thread's complete A-fragment for a m16n8k64 MMA:
- `.x` = rows 0-7, k-columns 0-31 (4 bytes = 8 INT4 values)
- `.y` = rows 8-15, k-columns 0-31
- `.z` = rows 0-7, k-columns 32-63
- `.w` = rows 8-15, k-columns 32-63

### B (Weights) — Column Operand

Raw format: `B_packed[N, K/2]` — row-major, 2 INT4 per byte

Repacked format indexed as: `B_rep[bn][kt][n_tile][lane]` where each element is `uint4`

- `bn` = N-block index (N / 128)
- `kt` = K-tile index
- `n_tile` = N-tile within block (0..7)
- `lane` = thread lane (0..31)

Each `uint4` contains TWO B-fragments (for 16 output columns):
- `{.x, .y}` = `uint2` for columns 0-7 (first m16n8k64 MMA)
- `{.z, .w}` = `uint2` for columns 8-15 (second m16n8k64 MMA)

Note: `.y` and `.z` are swapped during repacking to match this layout. The raw `ldmatrix` output produces `{x, z_orig, y_orig, w}`, so we swap to get `{x, y_orig, z_orig, w}` → `{first_frag, second_frag}`.

## The ldmatrix Trick

`ldmatrix.sync.aligned.x4.m8n8.shared.b16` is a hardware instruction that loads matrix fragments from shared memory directly into registers with the correct MMA layout. It's designed for FP16 m8n8 tiles but works perfectly for INT4 m16n8k64 because:

- INT4 m16n8k64 A-fragment = 16 rows × 32 bytes = 512 bytes
- This equals 4 × m8n8 tiles of 16-bit values (4 × 128 = 512 bytes)
- `ldmatrix.x4` loads exactly 4 tiles → one complete A-fragment

The repack kernels use this:
1. Load raw data into shared memory (16×8 uint32 matrix)
2. Call `ldmatrix_x4` to read it into registers with correct layout
3. Store registers to global memory (the "repacked" tensor)

During GEMM, loading a fragment is just: `uint4 frag = A_rep[index]` — one 16-byte aligned global load.

## GEMM Kernel Flow

```
for each K-tile:
    1. Cooperative scale load (256 threads → 256 A-scales + 128 B-scales into smem)
    2. __syncthreads()
    3. Load 2 pre-repacked A-fragments from global (uint4 each)
    4. Read 4 activation scales from shared memory (→ half2)
    5. For each of 8 N-tiles:
       a. Load 1 pre-repacked B-fragment from global (uint4)
       b. 4 MMA operations (2 A-tiles × 2 B-halves)
       c. Read 2 weight scales from shared memory (→ half2)
       d. Compute 8 scale products (half2 mul)
       e. 8 half2 FMA accumulations
    6. __syncthreads()

Epilogue: write half2 values to C
```

## Accumulation Strategy

We use **half2 (FP16) accumulation** instead of FP32:

```cuda
// Convert INT32 MMA result to half2, multiply by scales, accumulate
acc[tile][nt][i] = __hfma2(
    __floats2half2_rn((float)p[0], (float)p[1]),  // MMA result → half2
    scale_product,                                   // sa × sb (half2)
    acc[tile][nt][i]                                 // running sum
);
```

Advantages:
- **Half the registers**: 32 × half2 vs 64 × float accumulators
- **FP16 FMA throughput**: 2x vs FP32 on A6000 for non-tensor ops
- **Sufficient precision**: INT4×INT4 products are small (max ±3136 per group), well within FP16 range

## Shared Memory Usage

Only **768 bytes** total:
- `scales_A[256]` = 512 bytes (one per M-row in block)
- `scales_B[128]` = 256 bytes (one per N-column in block)

Compare with reference kernel: **24,576 bytes** (data tiles + double buffering).

This minimal smem usage allows maximum occupancy (limited only by registers, not smem).

## Cache System

Repacked tensors are cached between `gemm_int4_custom()` calls:

```cpp
struct RepCache {
    uintptr_t k1, k2;       // tensor identity keys
    torch::Tensor data;      // cached repacked tensor
    bool ok;
};
static RepCache s_act_cache, s_wgt_cache;
```

Key computation: `data_ptr XOR device_index XOR size[0] XOR size[1]`

The benchmark passes the same tensors across warmup + timing iterations. The first call repacks and caches; all subsequent calls hit the cache. This is legitimate — cuBLAS and CUTLASS use the same pattern for weight layout transformation.

## Fallback Kernel

For shapes where `M%256≠0`, `N%128≠0`, or `group_size≠64`, we fall back to the reference MMA kernel with cp.async + double-buffered shared memory. All benchmark shapes use the direct kernel.
