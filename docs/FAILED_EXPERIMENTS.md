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
For ff_down (K=12288), the splitK path allocated `[4, 4096, 3072] float32` = 192MB, launched a second reduction kernel, and used `float acc[2][8][8]` = 128 float registers (vs 64 half2 in v2). The memory allocation + reduction kernel overhead dominated GEMM time.

### Root cause 3: CTA swizzle added overhead
Extra index computation per block + potentially worse L2 access pattern.

**Lesson**: NEVER apply multiple untested optimizations simultaneously. Each change must be benchmarked independently. Register spilling is the #1 killer.

## Experiment 6: BM=128 with 8 warps (AT=1)
**Result: 239 TOPs (was 299) — 20% REGRESSION**

Halved BM to 128, reducing AT from 2 to 1 per warp. Goal: halve accumulator registers → more occupancy.

**Root cause**: Each warp now does 16 MMAs per K-tile instead of 32. The MMA-to-overhead ratio worsened because the overhead (scale loads, dequant) stayed similar while compute halved. Even though occupancy doubled (4 blocks/SM), the total MMA throughput per SM was lower.

**Lesson**: Reducing per-warp compute to increase occupancy is only worth it if the kernel is latency-bound (not compute-bound). At 48% of peak MMA, we're compute-bound.

## Experiment 7: Zero-sync kernel (transposed scales via L1)
**Result: 102 TOPs (was 299) — 3x REGRESSION**

Removed all shared memory for scales. Transposed scale arrays offline [rows, nkt] → [nkt, rows] for coalesced L1 reads. Read scales directly from global memory.

**Root cause**: Each thread now does 20+ global memory loads per K-tile for scales (4 A-scales + 16 B-scale reads across N-tiles). Even with L1 cache hits (~30 cycle latency), the instruction count overwhelmed the pipeline. Shared memory reads are ~5 cycles, __syncthreads is ~25 cycles. The extra loads cost far more than the sync they eliminated.

**Lesson**: Shared memory is 6× faster than L1 for this access pattern. The __syncthreads overhead is small compared to replacing 20 smem reads with 20 global reads.

## Experiment 8: __launch_bounds__(256, 2) on direct kernel
**Result: CORRECTNESS FAILURE (cosine 0.69-0.80)**

Added `__launch_bounds__(256, 2)` to tell the compiler to optimize for 2 blocks/SM.

**Root cause**: With max 128 registers (65536/256/2) and ~125 used, the compiler had only 3 spare registers. The constrained allocation changed the instruction schedule and caused computation errors (not just spilling — actual incorrect results).

**Lesson**: Even seemingly safe __launch_bounds__ (just 3 registers over current usage) can cause the compiler to produce incorrect code. Never trust __launch_bounds__ on register-pressure-sensitive kernels.

## Experiment 9: Hoist B-scale reads outside K-tile loop
**Result: 269 TOPs (was 323) — 17% REGRESSION**

Pre-read all B scale values (sb01_all[8], sb23_all[8]) into register arrays before the K-tile loop, since they're constant within a B-group.

**Root cause**: Added 16 half2 registers (sb01_all + sb23_all). From 125 to ~141 registers → dropped from 2 blocks/SM to 1 block/SM. The occupancy halving destroyed performance.

**Lesson**: The register budget is razor-thin at 125/128. Even 16 extra registers (seemingly small) causes a 2× occupancy drop. Any optimization that adds registers MUST be verified against the 128-register ceiling.

## Summary: What Works vs What Doesn't

### Works
- ldmatrix repacking (57 → 283 TOPs) ✓
- Double-buffered scales (283 → 297 TOPs) ✓
- A-fragment prefetching before sync ✓
- half2 FMA accumulation ✓
- Vectorized half2 epilogue stores ✓
- Per-layer weight group sizes (299 → 315 TOPs) ✓
- __shfl_sync for A scales ✓
- 4-warp kernel for small-N shapes (315 → 323 TOPs) ✓
- Optimal clipping in quantize.py ✓

### Doesn't Work (on this kernel)
- Increasing BLOCK_K with wrong SMEM_STRIDE ✗
- __launch_bounds__ without verifying register count ✗
- K-loop unrolling (code duplication → spilling) ✗
- B-fragment manual prefetching (compiler already does it) ✗
- SplitK with huge workspace allocation ✗
- CTA swizzle (overhead > benefit) ✗
- Removing shared memory for scales (L1 too slow) ✗
- Hoisting smem reads into registers (register pressure) ✗

### Rules
1. **One change at a time. Benchmark. Keep if score goes UP, revert if DOWN.**
2. **Register count is the #1 constraint. Stay below 128 or die.**
3. **Don't try to outsmart the compiler on unrolled loops.**
4. **Shared memory is much faster than L1 for broadcast patterns.**
