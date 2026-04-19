# Failed Experiments & Lessons

## Experiment 1: BLOCK_K=128 with SMEM_STRIDE=64 (v3)
**Result: 28 TOPs (was 57) — 2x REGRESSION**

Changed BLOCK_K from 64 to 128 to halve __syncthreads count. Used SMEM_STRIDE=64 (no padding).

**Root cause**: SMEM_STRIDE=64 creates **4-way bank conflicts** on shared memory fragment loads. With stride 64, rows 0,2,4,6 all map to the same banks. Every load_a_frag and load_b_frag call serializes 4-way.

**Lesson**: SMEM_STRIDE=48 (for BK=64) gives ZERO bank conflicts because gcd(48/4, 32)=4 distributes rows across all banks. Never use a stride that's a multiple of 128 bytes (32 banks × 4 bytes).

## Experiment 2: BLOCK_N=64 + __launch_bounds__(256, 4) (v4 attempt)
**Result: 22 TOPs (was 57) — 2.6x REGRESSION**

Reduced BLOCK_N to 64 for higher occupancy (4 blocks/SM). Used __launch_bounds__(256, 4) which forces max 64 registers.

**Root cause**: With 32 float accumulators + ~30 other registers = ~62 needed, the 64-register limit caused the compiler to **spill registers to local memory**. Each spill load/store goes through L1→L2→DRAM, destroying performance.

**Lesson**: Never use __launch_bounds__ without knowing the exact register count. Check with `-Xptxas -v` first. Spilling is catastrophic — worse than low occupancy.

## Experiment 3: K-loop unroll by 2
**Result: 227 TOPs (was 297) — 24% REGRESSION**

Duplicated the N-tile loop body to process 2 K-tiles per __syncthreads barrier.

**Root cause**: Duplicating the loop body doubled the code size and register pressure. The compiler couldn't keep all the temporaries in registers across both iterations → spilling. The reduced sync count (~4% improvement) was overwhelmed by spill overhead.

**Lesson**: Code duplication increases register pressure even if variables are "logically scoped" to each copy. The compiler's register allocator sees the unrolled code as one large block.

## Experiment 4: B-fragment prefetching in N-tile loop (v2.1)
**Result: 297 TOPs (was 297) — NO CHANGE**

Added `bf_next = B[...]` load at the start of each N-tile iteration, used `bf = bf_next` at the end.

**Root cause**: The `#pragma unroll` already makes all 8 N-tile iterations visible to the compiler. NVCC was already reordering the loads optimally. The explicit prefetch added 1 extra register (bf_next) with no benefit.

**Lesson**: Don't try to outsmart the compiler on fully-unrolled loops. `#pragma unroll` + the compiler's instruction scheduler already handles this.

## Experiment 5: v4 Agent (CTA swizzle + A-frag double-buffer + splitK)
**Result: 31 TOPs (was 297) — 9.4x REGRESSION**

Three "optimizations" applied simultaneously by an agent.

### Root cause 1: __launch_bounds__(256, 3) + extra registers
The agent added `uint4 cur_a[AT], nxt_a[AT]` (8 extra registers for A-fragment double-buffering) while using `__launch_bounds__(256, 3)` which caps at 85 registers. The v2 kernel already uses 125 registers. Result: **massive register spilling**.

### Root cause 2: SplitK allocated 192MB workspace
For ff_down (K=12288), the splitK path allocated `[4, 4096, 3072] float32` = 192MB, launched a second reduction kernel, and used `float acc[2][8][8]` = 128 float registers (vs 64 half2 in v2). The memory allocation + reduction kernel overhead dominated GEMM time. ff_down dropped from 303 → 12.5 TOPs.

### Root cause 3: CTA swizzle added overhead
Extra index computation per block + potentially worse L2 access pattern (the "optimization" assumed a specific access pattern that didn't match reality).

**Lesson**: NEVER apply multiple untested optimizations simultaneously. Each change must be benchmarked independently. Register spilling is the #1 killer — always verify register count before adding __launch_bounds__.

## Summary: What Works vs What Doesn't

### Works
- ldmatrix repacking (57 → 283 TOPs) ✓
- Double-buffered scales (283 → 297 TOPs) ✓
- A-fragment prefetching before sync (283 → 297 TOPs) ✓
- half2 FMA accumulation ✓
- Vectorized half2 epilogue stores ✓

### Doesn't Work (on this kernel)
- Increasing BLOCK_K with wrong SMEM_STRIDE ✗
- __launch_bounds__ without verifying register count ✗
- K-loop unrolling (code duplication → spilling) ✗
- B-fragment manual prefetching (compiler already does it) ✗
- SplitK with huge workspace allocation ✗
- CTA swizzle (overhead > benefit) ✗

### Rule
**One change at a time. Benchmark. Keep if score goes UP, revert if DOWN.**
