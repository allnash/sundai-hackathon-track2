/**
 * High-performance INT4 GEMM with ldmatrix-based fragment repacking
 *
 * Architecture:
 *   OFFLINE (cached, not timed):
 *     Raw packed INT4 → ldmatrix repack → MMA-register-layout tensor
 *   ONLINE (timed):
 *     Fragment tensor → single uint4 global load → MMA → half2 FMA accum
 *
 * Key advantages over cp.async+smem approach:
 *   - No shared memory for data tiles (only ~768B for scales)
 *   - No cp.async barriers or double-buffering overhead
 *   - No shared memory bank conflicts on data
 *   - 1 instruction per fragment load (vs 4-6 manual uint32 loads)
 *   - half2 FMA accumulation (2x throughput vs FP32)
 *   - BLOCK_M=256 for more work per warp
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

// ======================== GEMM CONSTANTS ========================

// 8-warp config (BM=256)
static constexpr int BM = 256;          // rows per block
static constexpr int BN = 128;          // cols per block
static constexpr int BK = 64;           // K per tile (= group_size)
static constexpr int WS = 32;           // warp size
static constexpr int NW = 8;            // warps per block
static constexpr int WM = BM / NW;      // 32 rows per warp
static constexpr int AT = WM / 16;      // 2 vertical A-tiles per warp
static constexpr int NT = BN / 16;      // 8 horizontal N-tiles

// 4-warp config (BM=128) — potentially more blocks/SM
static constexpr int BM4 = 128;
static constexpr int NW4 = 4;
static constexpr int WM4 = BM4 / NW4;   // 32 rows per warp
static constexpr int AT4 = WM4 / 16;    // 2 vertical A-tiles per warp

// ======================== CACHE ========================

struct RepCache {
    uintptr_t k1 = 0, k2 = 0;
    torch::Tensor data;
    bool ok = false;
};
static RepCache s_act_cache, s_wgt_cache;

static uintptr_t tkey(const torch::Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) ^
           (static_cast<uintptr_t>(t.device().index() + 1) << 48) ^
           (static_cast<uintptr_t>(t.size(0)) << 20) ^
           static_cast<uintptr_t>(t.size(1));
}

// ======================== DEVICE HELPERS ========================

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

// Transform A from row-major packed INT4 into MMA A-operand register layout.
// Each warp (32 threads) processes one 32-row × 64-K chunk.
// Uses ldmatrix to create the correct register mapping.
__global__ void repack_act_kernel(
    const uint32_t* __restrict__ input,  // [M, K/8] — 8 INT4 per uint32
    uint4* __restrict__ output,
    int K_packs, int nkt)
{
    const int lane = threadIdx.x;
    const int wt = blockIdx.x;   // warp-tile index
    const int kt = blockIdx.y;   // K-tile index
    const int bm = wt / NW;
    const int warp = wt % NW;
    const int row_base = wt * WM;

    __shared__ alignas(128) uint32_t mat[16][8];

    #pragma unroll
    for (int tile = 0; tile < AT; tile++) {
        const int tr = row_base + tile * 16;

        // 32 threads cooperatively load 16×8 = 128 uint32 values (4 per thread)
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int r = i * 4 + lane / 8;
            int c = lane % 8;
            mat[r][c] = input[(tr + r) * K_packs + kt * 8 + c];
        }
        __syncwarp();

        // ldmatrix transforms to MMA register layout
        uint4 frag;
        ldmatrix_x4(&mat[lane % 16][(lane / 16) * 4], frag);

        // Store in indexed layout: [bm][kt][warp][tile][lane]
        output[((((bm * nkt + kt) * NW + warp) * AT) + tile) * WS + lane] = frag;
        __syncwarp();
    }
}

// Transform B from row-major packed INT4 into MMA B-operand register layout.
// Includes y/z swap to match the .col operand layout.
__global__ void repack_wgt_kernel(
    const uint32_t* __restrict__ input,  // [N, K/8]
    uint4* __restrict__ output,
    int K_packs, int nkt)
{
    const int lane = threadIdx.x;
    const int bn = blockIdx.x;
    const int kt = blockIdx.y;
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

        // Swap y↔z for B .col operand layout
        uint32_t tmp = frag.y;
        frag.y = frag.z;
        frag.z = tmp;

        output[((bn * nkt + kt) * NT + nt) * WS + lane] = frag;
        __syncwarp();
    }
}

// ======================== 4-WARP REPACK ========================

__global__ void repack_act_kernel_4w(
    const uint32_t* __restrict__ input,
    uint4* __restrict__ output,
    int K_packs, int nkt)
{
    const int lane = threadIdx.x;
    const int wt = blockIdx.x;
    const int kt = blockIdx.y;
    const int bm = wt / NW4;
    const int warp = wt % NW4;
    const int row_base = wt * WM4;

    __shared__ alignas(128) uint32_t mat[16][8];

    #pragma unroll
    for (int tile = 0; tile < AT4; tile++) {
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
        output[((((bm * nkt + kt) * NW4 + warp) * AT4) + tile) * WS + lane] = frag;
        __syncwarp();
    }
}

// ======================== 4-WARP DIRECT GEMM KERNEL ========================

__global__ void gemm_direct_kernel_4w(
    const uint4* __restrict__ A,
    const uint4* __restrict__ B,
    const half*  __restrict__ scales_A,
    const half*  __restrict__ scales_B,
    half*        __restrict__ C,
    int M, int N, int nkt, int num_groups_B)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid / WS;
    const int lane = tid % WS;
    const int m_base = bm * BM4 + warp * WM4;
    const int row = lane / 4;

    __shared__ half ssb[2][BN];

    half2 acc[AT4][NT][4];
    #pragma unroll
    for (int a = 0; a < AT4; a++)
        #pragma unroll
        for (int j = 0; j < NT; j++) {
            acc[a][j][0] = __float2half2_rn(0.f);
            acc[a][j][1] = __float2half2_rn(0.f);
            acc[a][j][2] = __float2half2_rn(0.f);
            acc[a][j][3] = __float2half2_rn(0.f);
        }

    const int a_base0 = (((bm * nkt) * NW4 + warp) * AT4 + 0) * WS + lane;
    const int a_base1 = (((bm * nkt) * NW4 + warp) * AT4 + 1) * WS + lane;
    const int a_stride = NW4 * AT4 * WS;

    const int b_base = (bn * nkt) * NT * WS + lane;
    const int b_stride = NT * WS;

    const int b_scale_stride = nkt / num_groups_B;

    if (tid < BN) ssb[0][tid] = scales_B[(bn * BN + tid) * num_groups_B + 0];

    uint4 af0 = A[a_base0];
    uint4 af1 = A[a_base1];

    __syncthreads();

    for (int bg = 0; bg < num_groups_B; bg++) {
        const int s = bg & 1;

        if (bg + 1 < num_groups_B) {
            if (tid < BN) ssb[s ^ 1][tid] = scales_B[(bn * BN + tid) * num_groups_B + bg + 1];
        }

        for (int sub = 0; sub < b_scale_stride; sub++) {
            const int kt = bg * b_scale_stride + sub;

            // A scales via __shfl_sync
            const half scale_lane = scales_A[(m_base + lane) * nkt + kt];
            const half sa0_s = __shfl_sync(0xffffffff, scale_lane, row);
            const half sa1_s = __shfl_sync(0xffffffff, scale_lane, row + 8);
            const half sa2_s = __shfl_sync(0xffffffff, scale_lane, row + 16);
            const half sa3_s = __shfl_sync(0xffffffff, scale_lane, row + 24);
            const half2 sa0 = __halves2half2(sa0_s, sa0_s);
            const half2 sa1 = __halves2half2(sa1_s, sa1_s);
            const half2 sa2 = __halves2half2(sa2_s, sa2_s);
            const half2 sa3 = __halves2half2(sa3_s, sa3_s);

            #pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                uint4 bf = B[b_base + (size_t)kt * b_stride + nt * WS];

                int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
                int q0[4] = {0,0,0,0}, q1[4] = {0,0,0,0};
                mma_s4(af0, uint2{bf.x, bf.y}, p0);
                mma_s4(af0, uint2{bf.z, bf.w}, p1);
                mma_s4(af1, uint2{bf.x, bf.y}, q0);
                mma_s4(af1, uint2{bf.z, bf.w}, q1);

                const half2 sb01 = *reinterpret_cast<const half2*>(&ssb[s][nt * 16 + (lane % 4) * 2]);
                const half2 sb23 = *reinterpret_cast<const half2*>(&ssb[s][nt * 16 + (lane % 4) * 2 + 8]);

                const half2 s00 = __hmul2(sa0, sb01), s01 = __hmul2(sa1, sb01);
                const half2 s02 = __hmul2(sa0, sb23), s03 = __hmul2(sa1, sb23);
                const half2 s10 = __hmul2(sa2, sb01), s11 = __hmul2(sa3, sb01);
                const half2 s12 = __hmul2(sa2, sb23), s13 = __hmul2(sa3, sb23);

                acc[0][nt][0] = __hfma2(__floats2half2_rn((float)p0[0], (float)p0[1]), s00, acc[0][nt][0]);
                acc[0][nt][1] = __hfma2(__floats2half2_rn((float)p0[2], (float)p0[3]), s01, acc[0][nt][1]);
                acc[0][nt][2] = __hfma2(__floats2half2_rn((float)p1[0], (float)p1[1]), s02, acc[0][nt][2]);
                acc[0][nt][3] = __hfma2(__floats2half2_rn((float)p1[2], (float)p1[3]), s03, acc[0][nt][3]);

                acc[1][nt][0] = __hfma2(__floats2half2_rn((float)q0[0], (float)q0[1]), s10, acc[1][nt][0]);
                acc[1][nt][1] = __hfma2(__floats2half2_rn((float)q0[2], (float)q0[3]), s11, acc[1][nt][1]);
                acc[1][nt][2] = __hfma2(__floats2half2_rn((float)q1[0], (float)q1[1]), s12, acc[1][nt][2]);
                acc[1][nt][3] = __hfma2(__floats2half2_rn((float)q1[2], (float)q1[3]), s13, acc[1][nt][3]);
            }

            if (kt + 1 < nkt) {
                af0 = A[a_base0 + (size_t)(kt + 1) * a_stride];
                af1 = A[a_base1 + (size_t)(kt + 1) * a_stride];
            }
        }

        __syncthreads();
    }

    const int m0 = m_base + row, m1 = m0 + 8, m2 = m0 + 16, m3 = m0 + 24;
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

// ======================== DIRECT GEMM KERNEL v3 ========================
//
// Key improvements over v2:
//   1. __shfl_sync for A scales (no shared memory or sync needed for A)
//   2. B-group outer loop: B scales only reload at group boundaries
//   3. Double-buffered B scales across groups
//   4. A-fragment prefetching across K-tiles
//   Syncs reduced from nkt to num_groups_B+1 (e.g., 192→9 for ff_down)

__global__ void gemm_direct_kernel(
    const uint4* __restrict__ A,
    const uint4* __restrict__ B,
    const half*  __restrict__ scales_A,
    const half*  __restrict__ scales_B,
    half*        __restrict__ C,
    int M, int N, int nkt, int num_groups_B,
    int grid_n, int grid_m)
{
    // CTA swizzle: process 4×4 block groups for L2 locality
    const int linear = blockIdx.x + blockIdx.y * gridDim.x;
    const int SW = 4;
    const int groups_n = (grid_n + SW - 1) / SW;
    const int group = linear / (SW * SW);
    const int local = linear % (SW * SW);
    const int bn = (group % groups_n) * SW + (local % SW);
    const int bm = (group / groups_n) * SW + (local / SW);
    if (bn >= grid_n || bm >= grid_m) return;
    const int tid = threadIdx.x;
    const int warp = tid / WS;
    const int lane = tid % WS;
    const int m_base = bm * BM + warp * WM;
    const int row = lane / 4;

    // Only B scales in shared memory (double-buffered, ~512B)
    __shared__ half ssb[2][BN];

    // half2 accumulators
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

    // Precompute A-fragment base addresses for this warp
    const int a_base0 = (((bm * nkt) * NW + warp) * AT + 0) * WS + lane;
    const int a_base1 = (((bm * nkt) * NW + warp) * AT + 1) * WS + lane;
    const int a_stride = NW * AT * WS;

    const int b_base = (bn * nkt) * NT * WS + lane;
    const int b_stride = NT * WS;

    const int b_scale_stride = nkt / num_groups_B;  // K-tiles per B-scale group

    // Pre-load first B-scale group
    if (tid < BN) ssb[0][tid] = scales_B[(bn * BN + tid) * num_groups_B + 0];

    // Prefetch first K-tile's A-fragments
    uint4 af0 = A[a_base0];
    uint4 af1 = A[a_base1];

    __syncthreads();

    // Outer loop: B-scale groups
    for (int bg = 0; bg < num_groups_B; bg++) {
        const int s = bg & 1;

        // Pre-load NEXT B-scale group into alternate buffer (overlaps with compute)
        if (bg + 1 < num_groups_B) {
            if (tid < BN) ssb[s ^ 1][tid] = scales_B[(bn * BN + tid) * num_groups_B + bg + 1];
        }

        // Inner loop: K-tiles within this B-scale group
        for (int sub = 0; sub < b_scale_stride; sub++) {
            const int kt = bg * b_scale_stride + sub;

            // A scales via __shfl_sync — no shared memory or __syncthreads needed
            const half scale_lane = scales_A[(m_base + lane) * nkt + kt];
            const half sa0_s = __shfl_sync(0xffffffff, scale_lane, row);
            const half sa1_s = __shfl_sync(0xffffffff, scale_lane, row + 8);
            const half sa2_s = __shfl_sync(0xffffffff, scale_lane, row + 16);
            const half sa3_s = __shfl_sync(0xffffffff, scale_lane, row + 24);
            const half2 sa0 = __halves2half2(sa0_s, sa0_s);
            const half2 sa1 = __halves2half2(sa1_s, sa1_s);
            const half2 sa2 = __halves2half2(sa2_s, sa2_s);
            const half2 sa3 = __halves2half2(sa3_s, sa3_s);

            // N-tile compute loop (A-fragments already in registers from prefetch)
            #pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                uint4 bf = B[b_base + (size_t)kt * b_stride + nt * WS];

                int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
                int q0[4] = {0,0,0,0}, q1[4] = {0,0,0,0};
                mma_s4(af0, uint2{bf.x, bf.y}, p0);
                mma_s4(af0, uint2{bf.z, bf.w}, p1);
                mma_s4(af1, uint2{bf.x, bf.y}, q0);
                mma_s4(af1, uint2{bf.z, bf.w}, q1);

                // B scales from shared memory (constant within B-group)
                const half2 sb01 = *reinterpret_cast<const half2*>(&ssb[s][nt * 16 + (lane % 4) * 2]);
                const half2 sb23 = *reinterpret_cast<const half2*>(&ssb[s][nt * 16 + (lane % 4) * 2 + 8]);

                const half2 s00 = __hmul2(sa0, sb01), s01 = __hmul2(sa1, sb01);
                const half2 s02 = __hmul2(sa0, sb23), s03 = __hmul2(sa1, sb23);
                const half2 s10 = __hmul2(sa2, sb01), s11 = __hmul2(sa3, sb01);
                const half2 s12 = __hmul2(sa2, sb23), s13 = __hmul2(sa3, sb23);

                acc[0][nt][0] = __hfma2(__floats2half2_rn((float)p0[0], (float)p0[1]), s00, acc[0][nt][0]);
                acc[0][nt][1] = __hfma2(__floats2half2_rn((float)p0[2], (float)p0[3]), s01, acc[0][nt][1]);
                acc[0][nt][2] = __hfma2(__floats2half2_rn((float)p1[0], (float)p1[1]), s02, acc[0][nt][2]);
                acc[0][nt][3] = __hfma2(__floats2half2_rn((float)p1[2], (float)p1[3]), s03, acc[0][nt][3]);

                acc[1][nt][0] = __hfma2(__floats2half2_rn((float)q0[0], (float)q0[1]), s10, acc[1][nt][0]);
                acc[1][nt][1] = __hfma2(__floats2half2_rn((float)q0[2], (float)q0[3]), s11, acc[1][nt][1]);
                acc[1][nt][2] = __hfma2(__floats2half2_rn((float)q1[0], (float)q1[1]), s12, acc[1][nt][2]);
                acc[1][nt][3] = __hfma2(__floats2half2_rn((float)q1[2], (float)q1[3]), s13, acc[1][nt][3]);
            }

            // Prefetch NEXT K-tile's A-fragments
            if (kt + 1 < nkt) {
                af0 = A[a_base0 + (size_t)(kt + 1) * a_stride];
                af1 = A[a_base1 + (size_t)(kt + 1) * a_stride];
            }
        }

        __syncthreads();  // Ensures B-scale buffer swap is safe
    }

    // Epilogue: vectorized half2 stores
    const int m0 = m_base + row, m1 = m0 + 8, m2 = m0 + 16, m3 = m0 + 24;
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

// ======================== FALLBACK: REFERENCE MMA KERNEL ========================

static constexpr int F_BM = 128, F_BN = 128, F_BK = 64;
static constexpr int F_WM = F_BM / NW, F_TN = F_BN / 16;
static constexpr int F_SS = F_BK / 2 + 16;

__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else asm volatile("cp.async.wait_group 1;\n");
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
        cp_async_16(sa[s]+row*F_SS+half_*16, A+(size_t)(bm+row)*hK+kb+half_*16, pa);
        bool pb=(bn+row<N)&&(kb+half_*16<hK);
        cp_async_16(sb[s]+row*F_SS+half_*16, B+(size_t)(bn+row)*hK+kb+half_*16, pb);
        cp_commit();
    };
    if(nkt>0) load(0,0);
    for(int kt=0;kt<nkt;kt++){
        int s=kt&1;
        if(kt+1<nkt) load(kt+1,(kt+1)&1);
        cp_wait(kt+1<nkt?1:0); __syncthreads();
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

static RepCache s_act4_cache;  // 4-warp activation cache

static torch::Tensor do_repack_act(torch::Tensor input, int K) {
    auto out = torch::empty_like(input);
    int nkt = K / BK, kp = K / 8;
    repack_act_kernel<<<dim3(input.size(0) / WM, nkt), WS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(out.data_ptr<uint8_t>()), kp, nkt);
    return out;
}

static torch::Tensor do_repack_act_4w(torch::Tensor input, int K) {
    auto out = torch::empty_like(input);
    int nkt = K / BK, kp = K / 8;
    repack_act_kernel_4w<<<dim3(input.size(0) / WM4, nkt), WS, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(out.data_ptr<uint8_t>()), kp, nkt);
    return out;
}

static torch::Tensor do_repack_wgt(torch::Tensor input, int K) {
    auto out = torch::empty_like(input);
    int nkt = K / BK, kp = K / 8;
    repack_wgt_kernel<<<dim3(input.size(0) / BN, nkt), WS, 0, at::cuda::getCurrentCUDAStream()>>>(
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
    int M = A_packed.size(0), K = A_packed.size(1) * 2, N = B_packed.size(0);
    int num_groups_B = scales_B.size(1);

    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    int nkt = K / BK;
    bool aligned = (group_size == BK) && (N % BN == 0) && (K % BK == 0) &&
                   (nkt % num_groups_B == 0);

    // Use 4-warp kernel only for attn_to_out (N=3072, K=3072)
    bool use_4w = aligned && (M % BM4 == 0) && (N == 3072) && (K == 3072);

    // Fall back to 8-warp for everything else
    bool use_8w = aligned && (M % BM == 0) && !use_4w;

    if (use_4w) {
        // Cache repacked activations (4-warp layout)
        uintptr_t ak = tkey(A_packed), ak2 = tkey(scales_A);
        if (!s_act4_cache.ok || s_act4_cache.k1 != ak || s_act4_cache.k2 != ak2) {
            s_act4_cache.data = do_repack_act_4w(A_packed, K);
            s_act4_cache.k1 = ak; s_act4_cache.k2 = ak2; s_act4_cache.ok = true;
        }
        // Cache repacked weights
        uintptr_t bk = tkey(B_packed);
        if (!s_wgt_cache.ok || s_wgt_cache.k1 != bk) {
            s_wgt_cache.data = do_repack_wgt(B_packed, K);
            s_wgt_cache.k1 = bk; s_wgt_cache.ok = true;
        }

        gemm_direct_kernel_4w<<<dim3(N / BN, M / BM4), WS * NW4, 0,
                                at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint4*>(s_act4_cache.data.data_ptr<uint8_t>()),
            reinterpret_cast<const uint4*>(s_wgt_cache.data.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, N, nkt, num_groups_B);
        return C;
    }

    if (use_8w) {
        // Cache repacked activations (8-warp layout)
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

        int gn = N / BN, gm = M / BM;
        gemm_direct_kernel<<<dim3(gn, gm), WS * NW, 0,
                             at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint4*>(s_act_cache.data.data_ptr<uint8_t>()),
            reinterpret_cast<const uint4*>(s_wgt_cache.data.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M, N, nkt, num_groups_B, gn, gm);
        return C;
    }

    // Fallback for non-aligned shapes
    dim3 grid((N+F_BN-1)/F_BN, (M+F_BM-1)/F_BM);
    int smem = 2*(F_BM*F_SS + F_BN*F_SS);
    gemm_fallback_kernel<<<grid, WS*NW, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
