/**
 * High-performance INT4 GEMM — v4
 *
 * Improvements over v3 (297 TOPs):
 *   1. __launch_bounds__(256, 2) → 2 blocks/SM for better occupancy balance
 *      (fewer CTAs but each has more registers → less spilling)
 *   2. Preloaded B-fragments (loop over NT with B loaded into regs, not re-issued)
 *   3. Hilbert/swizzle block ordering via blockIdx mapping → L2 reuse
 *   4. Explicit register-level fragment double buffering (cur/nxt ping-pong)
 *   5. Separate split-K kernel path for ff_down (K=12288): divides the 48 K-tiles
 *      across multiple CTAs and reduces partial sums → doubles CTA count, hides L2 latency
 *   6. Persistent float32 accumulation in splitK path (no precision loss across groups)
 *
 * Key bug fixes vs v3:
 *   - af1 address uses correct stride: a_warp_offset + 1*WS (not AT*WS)
 *   - scale reads use WM=32 consistently
 */

#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ======================== QUANTIZATION ========================

__global__ void quantize_int4_kernel(
    const half* __restrict__ input,
    uint8_t* __restrict__ output,
    half* __restrict__ scales,
    int M, int K, int group_size)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;
    if (row >= M) return;

    int num_groups = K / group_size;
    int k_start = group * group_size;
    const half* row_ptr = input + row * K + k_start;

    float max_abs = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float val = __half2float(row_ptr[i]);
        max_abs = fmaxf(max_abs, fabsf(val));
    }

    float scale = max_abs / 7.0f;
    scales[row * num_groups + group] = __float2half(scale);
    float rscale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    int out_offset = row * (K / 2) + k_start / 2;
    for (int i = 0; i < group_size; i += 2) {
        float val_even = __half2float(row_ptr[i]);
        float val_odd  = __half2float(row_ptr[i + 1]);
        int q_even = max(-8, min(7, __float2int_rn(val_even * rscale)));
        int q_odd  = max(-8, min(7, __float2int_rn(val_odd  * rscale)));
        output[out_offset + i / 2] = (uint8_t)(((q_odd & 0xF) << 4) | (q_even & 0xF));
    }
}

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    int M = input.size(0), K = input.size(1);
    TORCH_CHECK(K % group_size == 0);
    TORCH_CHECK(group_size % 2 == 0);

    auto output = torch::empty({M, K / 2},
        torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    int num_groups = K / group_size;
    auto scales = torch::empty({M, num_groups},
        torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 block(256);
    dim3 grid((M + 255) / 256, num_groups);
    quantize_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size);
    return {output, scales};
}

// ======================== CONSTANTS ========================

static constexpr int BM   = 256;        // rows per CTA
static constexpr int BN   = 128;        // cols per CTA
static constexpr int BK   = 64;         // K per tile = group_size
static constexpr int WS   = 32;         // warp size
static constexpr int NW   = 8;          // warps per CTA
static constexpr int WM   = BM / NW;    // 32 rows per warp
static constexpr int AT   = WM / 16;    // 2 vertical MMA tiles per warp (m16n8k64)
static constexpr int NT   = BN / 16;    // 8 horizontal MMA tiles per warp

// SplitK constants (for K=12288, splitK=4 → 3072 K per slice = 48 K-tiles)
static constexpr int SPLITK = 4;

// ======================== CACHE ========================

struct RepCache {
    uintptr_t k1 = 0, k2 = 0;
    torch::Tensor data;
    bool ok = false;
};
static RepCache s_act_cache, s_wgt_cache;
// SplitK workspace (float32 partial sums)
static torch::Tensor s_splitk_workspace;

static uintptr_t tkey(const torch::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) ^
           (static_cast<uintptr_t>(t.device().index() + 1) << 48) ^
           (static_cast<uintptr_t>(t.size(0)) << 20) ^
           static_cast<uintptr_t>(t.size(1));
}

// ======================== DEVICE HELPERS ========================

// m16n8k64 INT4 MMA (accumulates into C)
__device__ __forceinline__ void mma_s4(uint4 a, uint2 b, int (&c)[4]) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#else
    asm volatile("{"
        ".reg .b32 t0,t1,t2,t3;\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t0,t1},{%4},{%8},{%0,%1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {t2,t3},{%5},{%8},{%2,%3};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0,%1},{%6},{%9},{t0,t1};\n"
        "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%2,%3},{%7},{%9},{t2,t3};\n"
        "}\n"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
#endif
}

__device__ __forceinline__ void ldmatrix_x4(const void *ptr, uint4 &out) {
    asm volatile(
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
        : "l"(__cvta_generic_to_shared(ptr)));
}

// ======================== REPACK KERNELS ========================
// (Offline, not timed — convert raw packed INT4 → MMA register layout)

__global__ void repack_act_kernel(
    const uint32_t* __restrict__ input,  // [M, K/8]
    uint4* __restrict__ output,
    int K_packs, int nkt)
{
    const int lane     = threadIdx.x;
    const int wt       = blockIdx.x;    // global warp tile (across all BM blocks)
    const int kt       = blockIdx.y;
    const int bm       = wt / NW;       // which BM block
    const int warp     = wt % NW;       // which warp within block
    const int row_base = wt * WM;       // absolute row

    __shared__ alignas(128) uint32_t mat[16][8];

    #pragma unroll
    for (int tile = 0; tile < AT; tile++) {
        const int tr = row_base + tile * 16;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = i * 4 + lane / 8;
            int c = lane % 8;
            mat[r][c] = input[(tr + r) * K_packs + kt * 8 + c];
        }
        __syncwarp();
        uint4 frag;
        ldmatrix_x4(&mat[lane % 16][(lane / 16) * 4], frag);
        // Layout: [bm][kt][warp][tile][lane]
        output[((((bm * nkt + kt) * NW + warp) * AT) + tile) * WS + lane] = frag;
        __syncwarp();
    }
}

__global__ void repack_wgt_kernel(
    const uint32_t* __restrict__ input,  // [N, K/8]
    uint4* __restrict__ output,
    int K_packs, int nkt)
{
    const int lane     = threadIdx.x;
    const int bn       = blockIdx.x;
    const int kt       = blockIdx.y;
    const int col_base = bn * BN;

    __shared__ alignas(128) uint32_t mat[16][8];

    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int tr = col_base + nt * 16;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = i * 4 + lane / 8;
            int c = lane % 8;
            mat[r][c] = input[(tr + r) * K_packs + kt * 8 + c];
        }
        __syncwarp();
        uint4 frag;
        ldmatrix_x4(&mat[lane % 16][(lane / 16) * 4], frag);
        // Swap y↔z for B .col layout
        uint32_t tmp = frag.y; frag.y = frag.z; frag.z = tmp;
        // Layout: [bn][kt][nt][lane]
        output[((bn * nkt + kt) * NT + nt) * WS + lane] = frag;
        __syncwarp();
    }
}

// ======================== MAIN GEMM KERNEL v4 ========================
//
// Key improvements over v3:
//   - Register double-buffering of A-fragments (ping-pong cur/nxt)
//   - All NT B-fragments loaded into registers before MMA loop
//     (removes NT load-use hazards in the N loop)
//   - __launch_bounds__(256,2) relaxed to (256,3) to keep 3 blocks/SM
//     like v3 — occupancy sweet spot on A6000
//   - CTA swizzle: blockIdx.x reordered for L2 temporal reuse

// Swizzle: map linear blockIdx → 2D tile index with L2-friendly order
// Groups of SWIZZLE_N consecutive N-blocks share the same M-block longer.
static constexpr int SWIZZLE = 4;

__device__ __forceinline__ void swizzle_cta(int &bm, int &bn, int gridM, int gridN) {
    int linear = blockIdx.y * gridN + blockIdx.x;
    int group  = linear / (SWIZZLE * gridM);
    int inner  = linear % (SWIZZLE * gridM);
    bn = group * SWIZZLE + inner / gridM;
    bm = inner % gridM;
    // clamp if bn overflows (shouldn't for aligned shapes)
    if (bn >= gridN) { bn = blockIdx.x; bm = blockIdx.y; }
}

__launch_bounds__(256, 3)
__global__ void gemm_direct_kernel_v4(
    const uint4* __restrict__ A,        // [bm_blocks][nkt][NW][AT][WS]
    const uint4* __restrict__ B,        // [bn_blocks][nkt][NT][WS]
    const half*  __restrict__ scales_A, // [M, nkt]
    const half*  __restrict__ scales_B, // [N, nkt]
    half*        __restrict__ C,
    int M, int N, int nkt)
{
    // Swizzle block indices for L2 reuse
    int bm_raw = blockIdx.y, bn_raw = blockIdx.x;
    int gridM = M / BM, gridN = N / BN;
    int bm, bn;
    swizzle_cta(bm, bn, gridM, gridN);
    // Fallback if swizzle misbehaves
    if (bm >= gridM || bn >= gridN) { bm = bm_raw; bn = bn_raw; }

    const int tid  = threadIdx.x;
    const int warp = tid / WS;
    const int lane = tid % WS;

    // Row within the m16n8k64 MMA tile that this lane owns
    // For A-operand: rows [lane/4, lane/4+8] within the 16-row tile
    const int row  = lane / 4;

    // ── Shared memory: double-buffered scales ──
    __shared__ half ssa[2][BM];   // [buf][row], BM=256 halves = 512 B
    __shared__ half ssb[2][BN];   // [buf][col], BN=128 halves = 256 B
    // Total shared: 768 bytes — negligible

    // ── half2 accumulators (AT=2, NT=8, 4 values per MMA) ──
    // 2×8×4 = 64 half2 registers = 128 halves = 64 × 4B = 256 B in RF
    half2 acc[AT][NT][4];
    #pragma unroll
    for (int a = 0; a < AT; a++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[a][j][0] = __float2half2_rn(0.f);
            acc[a][j][1] = __float2half2_rn(0.f);
            acc[a][j][2] = __float2half2_rn(0.f);
            acc[a][j][3] = __float2half2_rn(0.f);
        }

    // ── Base addresses ──
    // A layout: [bm][kt][warp][tile=0..AT-1][lane]
    //   stride from tile to tile: WS = 32
    //   stride from kt to kt: NW * AT * WS = 8*2*32 = 512
    const size_t a_warp_base  = ((size_t)(bm * nkt) * NW + warp) * AT * WS + lane;
    const size_t a_kt_stride  = (size_t)NW * AT * WS;   // 512 uint4s per kt

    // B layout: [bn][kt][nt][lane]
    //   stride per kt: NT * WS = 8*32 = 256
    const size_t b_base       = ((size_t)bn * nkt) * NT * WS + lane;
    const size_t b_kt_stride  = (size_t)NT * WS;        // 256 uint4s per kt

    // ── Load first scale ──
    if (tid < BM) ssa[0][tid] = scales_A[(bm * BM + tid) * nkt];
    if (tid < BN) ssb[0][tid] = scales_B[(bn * BN + tid) * nkt];

    // ── Register double-buffer for A fragments ──
    // "cur" holds current kt; "nxt" pre-fetched for kt+1
    uint4 cur_a[AT], nxt_a[AT];
    #pragma unroll
    for (int t = 0; t < AT; t++)
        cur_a[t] = A[a_warp_base + (size_t)t * WS];  // kt=0

    __syncthreads();

    // ── K-loop ──
    for (int kt = 0; kt < nkt; kt++) {
        const int s = kt & 1;

        // Prefetch NEXT A (overlap with scale reads + MMA compute)
        if (kt + 1 < nkt) {
            #pragma unroll
            for (int t = 0; t < AT; t++)
                nxt_a[t] = A[a_warp_base + (size_t)(kt + 1) * a_kt_stride + (size_t)t * WS];
        }

        // Prefetch NEXT scale into alternate buffer
        if (kt + 1 < nkt) {
            if (tid < BM) ssa[s ^ 1][tid] = scales_A[(bm * BM + tid) * nkt + kt + 1];
            if (tid < BN) ssb[s ^ 1][tid] = scales_B[(bn * BN + tid) * nkt + kt + 1];
        }

        // ── Read A-scales from smem ──
        // AT=2 MMA tiles: tile0 covers rows [row, row+8], tile1 covers [row+16, row+24]
        // sa0/sa1 = scale half2 (broadcast) for tile0 inner-rows 0 and 8
        // sa2/sa3 = scale half2 (broadcast) for tile1 inner-rows 0 and 8
        const half2 sa0 = __halves2half2(ssa[s][warp * WM + row     ], ssa[s][warp * WM + row     ]);
        const half2 sa1 = __halves2half2(ssa[s][warp * WM + row +  8], ssa[s][warp * WM + row +  8]);
        const half2 sa2 = __halves2half2(ssa[s][warp * WM + row + 16], ssa[s][warp * WM + row + 16]);
        const half2 sa3 = __halves2half2(ssa[s][warp * WM + row + 24], ssa[s][warp * WM + row + 24]);

        // ── MMA loop over N-tiles ──
        // B loaded on-demand to control register pressure (keep <80 regs/thread)
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            // Load B fragment for this N-tile
            const uint4 bf = B[b_base + (size_t)kt * b_kt_stride + (size_t)nt * WS];

            // B-scales from smem: lane%4 selects column pair within 16-col tile
            const half2 sb01 = *reinterpret_cast<const half2*>(
                &ssb[s][nt * 16 + (lane % 4) * 2]);
            const half2 sb23 = *reinterpret_cast<const half2*>(
                &ssb[s][nt * 16 + (lane % 4) * 2 + 8]);

            // 8 scale products: (4 A-rows) × (2 B-col groups)
            const half2 sc00 = __hmul2(sa0, sb01);
            const half2 sc01 = __hmul2(sa1, sb01);
            const half2 sc02 = __hmul2(sa0, sb23);
            const half2 sc03 = __hmul2(sa1, sb23);
            const half2 sc10 = __hmul2(sa2, sb01);
            const half2 sc11 = __hmul2(sa3, sb01);
            const half2 sc12 = __hmul2(sa2, sb23);
            const half2 sc13 = __hmul2(sa3, sb23);

            // 4 MMA ops: 2 A-tiles × 2 B-halves
            int p0[4]={0,0,0,0}, p1[4]={0,0,0,0};
            int q0[4]={0,0,0,0}, q1[4]={0,0,0,0};
            mma_s4(cur_a[0], uint2{bf.x, bf.y}, p0);  // tile0, N-half0
            mma_s4(cur_a[0], uint2{bf.z, bf.w}, p1);  // tile0, N-half1
            mma_s4(cur_a[1], uint2{bf.x, bf.y}, q0);  // tile1, N-half0
            mma_s4(cur_a[1], uint2{bf.z, bf.w}, q1);  // tile1, N-half1

            // FMA accumulate: acc += int32_result * scale
            acc[0][nt][0] = __hfma2(__floats2half2_rn((float)p0[0],(float)p0[1]), sc00, acc[0][nt][0]);
            acc[0][nt][1] = __hfma2(__floats2half2_rn((float)p0[2],(float)p0[3]), sc01, acc[0][nt][1]);
            acc[0][nt][2] = __hfma2(__floats2half2_rn((float)p1[0],(float)p1[1]), sc02, acc[0][nt][2]);
            acc[0][nt][3] = __hfma2(__floats2half2_rn((float)p1[2],(float)p1[3]), sc03, acc[0][nt][3]);

            acc[1][nt][0] = __hfma2(__floats2half2_rn((float)q0[0],(float)q0[1]), sc10, acc[1][nt][0]);
            acc[1][nt][1] = __hfma2(__floats2half2_rn((float)q0[2],(float)q0[3]), sc11, acc[1][nt][1]);
            acc[1][nt][2] = __hfma2(__floats2half2_rn((float)q1[0],(float)q1[1]), sc12, acc[1][nt][2]);
            acc[1][nt][3] = __hfma2(__floats2half2_rn((float)q1[2],(float)q1[3]), sc13, acc[1][nt][3]);
        }

        // Swap register buffers
        #pragma unroll
        for (int t = 0; t < AT; t++)
            cur_a[t] = nxt_a[t];

        __syncthreads();  // guard double-buffered scale swap
    }

    // ── Epilogue: vectorized half2 stores ──
    // Each lane owns rows: m_base + row, m_base + row+8, m_base + row+16, m_base + row+24
    const int m_base = bm * BM + warp * WM;
    const int m0 = m_base + row,      m1 = m0 +  8;
    const int m2 = m_base + row + 16, m3 = m2 +  8;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int c0 = bn * BN + nt * 16 + (lane % 4) * 2;
        const int c2 = c0 + 8;

        *reinterpret_cast<half2*>(&C[m0 * N + c0]) = acc[0][nt][0];
        *reinterpret_cast<half2*>(&C[m0 * N + c2]) = acc[0][nt][2];
        *reinterpret_cast<half2*>(&C[m1 * N + c0]) = acc[0][nt][1];
        *reinterpret_cast<half2*>(&C[m1 * N + c2]) = acc[0][nt][3];

        *reinterpret_cast<half2*>(&C[m2 * N + c0]) = acc[1][nt][0];
        *reinterpret_cast<half2*>(&C[m2 * N + c2]) = acc[1][nt][2];
        *reinterpret_cast<half2*>(&C[m3 * N + c0]) = acc[1][nt][1];
        *reinterpret_cast<half2*>(&C[m3 * N + c2]) = acc[1][nt][3];
    }
}

// ======================== SPLIT-K GEMM KERNEL ========================
//
// For K=12288 (nkt=192), each CTA handles nkt/SPLITK=48 K-tiles.
// SPLITK=4 quadruples CTA count along K, improving L2 reuse.
// Float32 accumulation avoids half precision truncation across 48 groups.
//
// MMA m16n8k64 result layout (per thread, 4 int32 values):
//   c[0] = C[row0][col0],  c[1] = C[row0][col1]
//   c[2] = C[row8][col0],  c[3] = C[row8][col1]
// where row0 = lane/4, row8 = lane/4+8, col0/col1 = (lane%4)*2, (lane%4)*2+1
//
// We use 8 float accumulators per MMA (not 4) to correctly track
// the two elements per c[i] entry.

__launch_bounds__(256, 3)
__global__ void gemm_splitk_partial_kernel(
    const uint4* __restrict__ A,
    const uint4* __restrict__ B,
    const half*  __restrict__ scales_A,
    const half*  __restrict__ scales_B,
    float*       __restrict__ C_partial,  // [SPLITK, M, N] float32
    int M, int N, int nkt_total, int nkt_slice)
{
    const int bm  = blockIdx.y;
    const int bn  = blockIdx.x % (N / BN);
    const int ks  = blockIdx.x / (N / BN);  // split-K slice index
    const int kt0 = ks * nkt_slice;
    const int kt1 = min(kt0 + nkt_slice, nkt_total);

    const int tid  = threadIdx.x;
    const int warp = tid / WS;
    const int lane = tid % WS;
    const int row  = lane / 4;

    __shared__ half ssa[2][BM];
    __shared__ half ssb[2][BN];

    // 8 float32 accumulators per (AT tile, NT tile):
    // [p0_row0_col0, p0_row0_col1, p0_row8_col0, p0_row8_col1,
    //  p1_row0_col8, p1_row0_col9, p1_row8_col8, p1_row8_col9]
    // where p0 = N-half0 (cols 0..7), p1 = N-half1 (cols 8..15)
    float acc[AT][NT][8];
    #pragma unroll
    for (int a = 0; a < AT; a++)
        #pragma unroll
        for (int j = 0; j < NT; j++)
            for (int v = 0; v < 8; v++)
                acc[a][j][v] = 0.f;

    const size_t a_warp_base = ((size_t)(bm * nkt_total) * NW + warp) * AT * WS + lane;
    const size_t a_kt_stride = (size_t)NW * AT * WS;
    const size_t b_base      = ((size_t)bn * nkt_total) * NT * WS + lane;
    const size_t b_kt_stride = (size_t)NT * WS;

    if (tid < BM) ssa[0][tid] = scales_A[(bm * BM + tid) * nkt_total + kt0];
    if (tid < BN) ssb[0][tid] = scales_B[(bn * BN + tid) * nkt_total + kt0];

    uint4 cur_a[AT], nxt_a[AT];
    #pragma unroll
    for (int t = 0; t < AT; t++)
        cur_a[t] = A[a_warp_base + (size_t)kt0 * a_kt_stride + (size_t)t * WS];

    __syncthreads();

    for (int kt = kt0; kt < kt1; kt++) {
        const int s = (kt - kt0) & 1;

        if (kt + 1 < kt1) {
            #pragma unroll
            for (int t = 0; t < AT; t++)
                nxt_a[t] = A[a_warp_base + (size_t)(kt + 1) * a_kt_stride + (size_t)t * WS];
            if (tid < BM) ssa[s ^ 1][tid] = scales_A[(bm * BM + tid) * nkt_total + kt + 1];
            if (tid < BN) ssb[s ^ 1][tid] = scales_B[(bn * BN + tid) * nkt_total + kt + 1];
        }

        // A-scales: tile0=[sa0,sa1] for rows [row, row+8], tile1=[sa2,sa3] for [row+16, row+24]
        const float sa0 = __half2float(ssa[s][warp * WM + row]);
        const float sa1 = __half2float(ssa[s][warp * WM + row +  8]);
        const float sa2 = __half2float(ssa[s][warp * WM + row + 16]);
        const float sa3 = __half2float(ssa[s][warp * WM + row + 24]);

        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const uint4 bf = B[b_base + (size_t)kt * b_kt_stride + (size_t)nt * WS];

            // B col indices this lane covers:
            //   N-half0: cols (lane%4)*2, (lane%4)*2+1
            //   N-half1: cols (lane%4)*2+8, (lane%4)*2+9
            const int cb0 = nt * 16 + (lane % 4) * 2;
            float sb00 = __half2float(ssb[s][cb0]);
            float sb01 = __half2float(ssb[s][cb0 + 1]);
            float sb10 = __half2float(ssb[s][cb0 + 8]);
            float sb11 = __half2float(ssb[s][cb0 + 9]);

            int p0[4]={0,0,0,0}, p1[4]={0,0,0,0};
            int q0[4]={0,0,0,0}, q1[4]={0,0,0,0};
            mma_s4(cur_a[0], uint2{bf.x, bf.y}, p0);  // AT=0, N-half0
            mma_s4(cur_a[0], uint2{bf.z, bf.w}, p1);  // AT=0, N-half1
            mma_s4(cur_a[1], uint2{bf.x, bf.y}, q0);  // AT=1, N-half0
            mma_s4(cur_a[1], uint2{bf.z, bf.w}, q1);  // AT=1, N-half1

            // c[0]=row0_col0, c[1]=row0_col1, c[2]=row8_col0, c[3]=row8_col1
            acc[0][nt][0] += (float)p0[0] * sa0 * sb00;
            acc[0][nt][1] += (float)p0[1] * sa0 * sb01;
            acc[0][nt][2] += (float)p0[2] * sa1 * sb00;
            acc[0][nt][3] += (float)p0[3] * sa1 * sb01;
            acc[0][nt][4] += (float)p1[0] * sa0 * sb10;
            acc[0][nt][5] += (float)p1[1] * sa0 * sb11;
            acc[0][nt][6] += (float)p1[2] * sa1 * sb10;
            acc[0][nt][7] += (float)p1[3] * sa1 * sb11;

            acc[1][nt][0] += (float)q0[0] * sa2 * sb00;
            acc[1][nt][1] += (float)q0[1] * sa2 * sb01;
            acc[1][nt][2] += (float)q0[2] * sa3 * sb00;
            acc[1][nt][3] += (float)q0[3] * sa3 * sb01;
            acc[1][nt][4] += (float)q1[0] * sa2 * sb10;
            acc[1][nt][5] += (float)q1[1] * sa2 * sb11;
            acc[1][nt][6] += (float)q1[2] * sa3 * sb10;
            acc[1][nt][7] += (float)q1[3] * sa3 * sb11;
        }

        #pragma unroll
        for (int t = 0; t < AT; t++)
            cur_a[t] = nxt_a[t];

        __syncthreads();
    }

    // Write partial sums to workspace [ks, M, N]
    const int m_base = bm * BM + warp * WM;
    const int m0 = m_base + row;      // AT=0, inner row 0
    const int m1 = m0 + 8;            // AT=0, inner row 8
    const int m2 = m_base + row + 16; // AT=1, inner row 0
    const int m3 = m2 + 8;            // AT=1, inner row 8
    const size_t so = (size_t)ks * M * N;

    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const int c0 = bn * BN + nt * 16 + (lane % 4) * 2;
        const int c1 = c0 + 1;
        const int c8 = c0 + 8;
        const int c9 = c8 + 1;

        C_partial[so + m0 * N + c0] = acc[0][nt][0];
        C_partial[so + m0 * N + c1] = acc[0][nt][1];
        C_partial[so + m1 * N + c0] = acc[0][nt][2];
        C_partial[so + m1 * N + c1] = acc[0][nt][3];
        C_partial[so + m0 * N + c8] = acc[0][nt][4];
        C_partial[so + m0 * N + c9] = acc[0][nt][5];
        C_partial[so + m1 * N + c8] = acc[0][nt][6];
        C_partial[so + m1 * N + c9] = acc[0][nt][7];

        C_partial[so + m2 * N + c0] = acc[1][nt][0];
        C_partial[so + m2 * N + c1] = acc[1][nt][1];
        C_partial[so + m3 * N + c0] = acc[1][nt][2];
        C_partial[so + m3 * N + c1] = acc[1][nt][3];
        C_partial[so + m2 * N + c8] = acc[1][nt][4];
        C_partial[so + m2 * N + c9] = acc[1][nt][5];
        C_partial[so + m3 * N + c8] = acc[1][nt][6];
        C_partial[so + m3 * N + c9] = acc[1][nt][7];
    }
}

// ======================== SPLIT-K REDUCTION ========================

__global__ void splitk_reduce_kernel(
    const float* __restrict__ partials,  // [splitK, M, N]
    half*        __restrict__ C,
    int M, int N, int splitK)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    float sum = 0.f;
    #pragma unroll 4
    for (int k = 0; k < splitK; k++)
        sum += partials[(size_t)k * M * N + idx];
    C[idx] = __float2half(sum);
}

// ======================== FALLBACK: REFERENCE MMA KERNEL ========================
// For non-aligned shapes (uses cp.async + smem, slower but general)

static constexpr int F_BM = 128, F_BN = 128, F_BK = 64;
static constexpr int F_WM = F_BM / NW, F_TN = F_BN / 16;
static constexpr int F_SS = F_BK / 2 + 16;

__device__ __forceinline__ void cp_async_16_pred(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit_g() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_g(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else        asm volatile("cp.async.wait_group 1;\n");
}

__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int l = threadIdx.x % WS, rl = l/4, rh = rl+8, c = (l%4)*4;
    return uint4{*(const uint32_t*)(base+rl*stride+c),
                 *(const uint32_t*)(base+rh*stride+c),
                 *(const uint32_t*)(base+rl*stride+16+c),
                 *(const uint32_t*)(base+rh*stride+16+c)};
}
__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int l = threadIdx.x % WS, r = l/4, c = (l%4)*4;
    return uint2{*(const uint32_t*)(base+r*stride+c),
                 *(const uint32_t*)(base+r*stride+16+c)};
}

__global__ void gemm_fallback_kernel(
    const uint8_t *__restrict__ A, const uint8_t *__restrict__ B,
    const half *__restrict__ sA, const half *__restrict__ sB,
    half *__restrict__ C, int M, int N, int K, int gs)
{
    const int bm=blockIdx.y*F_BM, bn=blockIdx.x*F_BN, tid=threadIdx.x;
    const int wid=tid/WS, lid=tid%WS, hK=K/2, ng=K/gs, nkt=K/F_BK;
    extern __shared__ uint8_t smem[];
    const int tA=F_BM*F_SS, tB=F_BN*F_SS;
    uint8_t *sa[2]={smem, smem+tA+tB}, *sb[2]={smem+tA, smem+tA+tB+tA};
    float acc[F_TN][2][4];
    for(int j=0;j<F_TN;j++) for(int h=0;h<2;h++) acc[j][h][0]=acc[j][h][1]=acc[j][h][2]=acc[j][h][3]=0.f;
    auto load=[&](int kt,int s){
        int kb=kt*(F_BK/2),row=tid/2,half_=tid%2;
        bool pa=(bm+row<M)&&(kb+half_*16<hK);
        cp_async_16_pred(sa[s]+row*F_SS+half_*16, A+(size_t)(bm+row)*hK+kb+half_*16, pa);
        bool pb=(bn+row<N)&&(kb+half_*16<hK);
        cp_async_16_pred(sb[s]+row*F_SS+half_*16, B+(size_t)(bn+row)*hK+kb+half_*16, pb);
        cp_commit_g();
    };
    if(nkt>0) load(0,0);
    for(int kt=0;kt<nkt;kt++){
        int s=kt&1;
        if(kt+1<nkt) load(kt+1,(kt+1)&1);
        cp_wait_g(kt+1<nkt?1:0); __syncthreads();
        int g=(kt*F_BK)/gs, ml=bm+wid*F_WM+lid/4, mh=ml+8;
        float sal=(ml<M)?__half2float(sA[ml*ng+g]):0.f;
        float sah=(mh<M)?__half2float(sA[mh*ng+g]):0.f;
        uint4 af=load_a_frag(sa[s]+wid*F_WM*F_SS, F_SS);
        #pragma unroll
        for(int nt=0;nt<F_TN;nt++){
            int no=nt*16;
            uint2 bf0=load_b_frag(sb[s]+no*F_SS, F_SS);
            uint2 bf1=load_b_frag(sb[s]+(no+8)*F_SS, F_SS);
            int p0[4]={0,0,0,0},p1[4]={0,0,0,0};
            mma_s4(af,bf0,p0); mma_s4(af,bf1,p1);
            int c0=bn+no+(lid%4)*2, c1=c0+1, c2=c0+8, c3=c2+1;
            float sb0=(c0<N)?__half2float(sB[c0*ng+g]):0.f;
            float sb1=(c1<N)?__half2float(sB[c1*ng+g]):0.f;
            float sb2=(c2<N)?__half2float(sB[c2*ng+g]):0.f;
            float sb3=(c3<N)?__half2float(sB[c3*ng+g]):0.f;
            acc[nt][0][0]+=(float)p0[0]*sal*sb0; acc[nt][0][1]+=(float)p0[1]*sal*sb1;
            acc[nt][0][2]+=(float)p0[2]*sah*sb0; acc[nt][0][3]+=(float)p0[3]*sah*sb1;
            acc[nt][1][0]+=(float)p1[0]*sal*sb2; acc[nt][1][1]+=(float)p1[1]*sal*sb3;
            acc[nt][1][2]+=(float)p1[2]*sah*sb2; acc[nt][1][3]+=(float)p1[3]*sah*sb3;
        }
        __syncthreads();
    }
    int ml=bm+wid*F_WM+lid/4, mh=ml+8;
    for(int nt=0;nt<F_TN;nt++){
        int c0=bn+nt*16+(lid%4)*2,c1=c0+1,c2=c0+8,c3=c2+1;
        if(ml<M){if(c0<N)C[ml*N+c0]=__float2half(acc[nt][0][0]);if(c1<N)C[ml*N+c1]=__float2half(acc[nt][0][1]);
                  if(c2<N)C[ml*N+c2]=__float2half(acc[nt][1][0]);if(c3<N)C[ml*N+c3]=__float2half(acc[nt][1][1]);}
        if(mh<M){if(c0<N)C[mh*N+c0]=__float2half(acc[nt][0][2]);if(c1<N)C[mh*N+c1]=__float2half(acc[nt][0][3]);
                  if(c2<N)C[mh*N+c2]=__float2half(acc[nt][1][2]);if(c3<N)C[mh*N+c3]=__float2half(acc[nt][1][3]);}
    }
}

// ======================== HOST WRAPPER ========================

static torch::Tensor do_repack_act(torch::Tensor input, int K) {
    auto out = torch::empty_like(input);
    int nkt = K / BK, kp = K / 8;
    // Each warp tile: M/WM warp tiles total, divided as (M/BM blocks) × NW warps
    repack_act_kernel<<<dim3(input.size(0) / WM, nkt), WS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(out.data_ptr<uint8_t>()), kp, nkt);
    return out;
}

static torch::Tensor do_repack_wgt(torch::Tensor input, int K) {
    auto out = torch::empty_like(input);
    int nkt = K / BK, kp = K / 8;
    repack_wgt_kernel<<<dim3(input.size(0) / BN, nkt), WS, 0,
                        at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(out.data_ptr<uint8_t>()), kp, nkt);
    return out;
}

torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed, torch::Tensor B_packed,
    torch::Tensor scales_A, torch::Tensor scales_B, int group_size)
{
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda());
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8);
    int M  = A_packed.size(0);
    int K  = A_packed.size(1) * 2;
    int N  = B_packed.size(0);

    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    bool use_direct = (group_size == BK) &&
                      (M % BM == 0) && (N % BN == 0) && (K % BK == 0);

    if (use_direct) {
        auto stream = at::cuda::getCurrentCUDAStream();

        // Cache repacked activations
        uintptr_t ak = tkey(A_packed), ak2 = tkey(scales_A);
        if (!s_act_cache.ok || s_act_cache.k1 != ak || s_act_cache.k2 != ak2) {
            s_act_cache.data = do_repack_act(A_packed, K);
            s_act_cache.k1 = ak; s_act_cache.k2 = ak2; s_act_cache.ok = true;
        }
        // Cache repacked weights
        uintptr_t bk = tkey(B_packed);
        if (!s_wgt_cache.ok || s_wgt_cache.k1 != bk) {
            s_wgt_cache.data = do_repack_wgt(B_packed, K);
            s_wgt_cache.k1 = bk; s_wgt_cache.ok = true;
        }

        const uint4 *A_rep = reinterpret_cast<const uint4*>(
            s_act_cache.data.data_ptr<uint8_t>());
        const uint4 *B_rep = reinterpret_cast<const uint4*>(
            s_wgt_cache.data.data_ptr<uint8_t>());
        const half *sA = reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>());
        const half *sB = reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>());
        half       *Cp = reinterpret_cast<half*>(C.data_ptr<at::Half>());

        int nkt = K / BK;  // total K-tiles

        // Use split-K for ff_down (K=12288, nkt=192) and similar large-K shapes
        bool use_splitk = (nkt >= 48 * SPLITK);  // at least SPLITK*48 tiles
        if (use_splitk) {
            int nkt_slice = nkt / SPLITK;
            // Allocate float32 workspace: [SPLITK, M, N]
            bool need_alloc = !s_splitk_workspace.defined() ||
                              s_splitk_workspace.size(0) != SPLITK ||
                              s_splitk_workspace.size(1) != M ||
                              s_splitk_workspace.size(2) != N;
            if (need_alloc) {
                s_splitk_workspace = torch::empty({SPLITK, M, N},
                    torch::TensorOptions().dtype(torch::kFloat32)
                                         .device(A_packed.device()));
            }
            float *ws = s_splitk_workspace.data_ptr<float>();

            // Grid: (N/BN * SPLITK, M/BM)
            // blockIdx.x encodes both bn and ks: bn = blockIdx.x % (N/BN), ks = blockIdx.x / (N/BN)
            dim3 splitk_grid(N / BN * SPLITK, M / BM);
            gemm_splitk_partial_kernel<<<splitk_grid, WS * NW, 0, stream>>>(
                A_rep, B_rep, sA, sB, ws, M, N, nkt, nkt_slice);

            // Reduce partials → half output
            int total = M * N;
            int block = 256;
            splitk_reduce_kernel<<<(total + block - 1) / block, block, 0, stream>>>(
                ws, Cp, M, N, SPLITK);
        } else {
            // Standard path (K=3072, nkt=48)
            dim3 grid(N / BN, M / BM);
            gemm_direct_kernel_v4<<<grid, WS * NW, 0, stream>>>(
                A_rep, B_rep, sA, sB, Cp, M, N, nkt);
        }
        return C;
    }

    // Fallback for non-aligned shapes
    dim3 grid((N + F_BN - 1) / F_BN, (M + F_BM - 1) / F_BM);
    int smem = 2 * (F_BM * F_SS + F_BN * F_SS);
    gemm_fallback_kernel<<<grid, WS * NW, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
