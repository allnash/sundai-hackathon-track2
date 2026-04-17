/**
 * MMA-based INT4 GEMM Kernel (Starting Code)
 *
 * Computes C[M,N] = A[M,K] @ B[N,K]^T using Tensor Core MMA instructions
 * on the standard packed INT4 format (uint8, low nibble=even, high nibble=odd).
 *
 * Key techniques demonstrated:
 *   - mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
 *   - cp.async.ca.shared.global for async prefetching
 *   - Double-buffered shared memory
 *   - Direct register packing from shared memory
 *
 * To use: copy both functions (gemm_int4_kernel + gemm_int4_custom) into
 * your kernel.cu, replacing the naive GEMM implementation.
 *
 * Requires SM >= 80 (Ampere). SM75 fallback via decomposed m8n8k32.
 */

#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>


// ---- Configuration ----
static constexpr int BLOCK_M   = 128;
static constexpr int BLOCK_N   = 128;
static constexpr int BLOCK_K   = 64;  // one quantization group per K-step
static constexpr int WARP_SZ   = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M    = BLOCK_M / NUM_WARPS;  // 16
static constexpr int TILES_N   = BLOCK_N / 16;         // 8 (16-col tiles)

// Shared memory stride: K/2 bytes + padding (must be 16-byte aligned for cp.async)
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;   // 48 bytes per row


// ---- MMA wrapper: m16n8k64 INT4×INT4 → INT32 ----
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


// ---- cp.async: 16-byte async global→shared copy ----
__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit()  { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait(int n) {
    if (n == 0) asm volatile("cp.async.wait_group 0;\n");
    else        asm volatile("cp.async.wait_group 1;\n");
}


// ---- Load MMA A-fragment directly from shared memory ----
// A is 16×64 INT4 in row-major packed format (16 rows × 32 bytes).
// MMA register mapping for m16n8k64.row.s4:
//   Thread T: groupID = T/4 (row pair: groupID and groupID+8)
//             localID = T%4 (column group: 8 consecutive INT4 starting at localID*8)
//   a.x = A[groupID,     localID*4 .. localID*4+3]  (4 bytes = 8 INT4, k=0..31 half)
//   a.y = A[groupID+8,   localID*4 .. localID*4+3]
//   a.z = A[groupID,     16+localID*4 .. 16+localID*4+3]  (k=32..63 half)
//   a.w = A[groupID+8,   16+localID*4 .. 16+localID*4+3]
__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4;          // 0-7
    int row_hi = row_lo + 8;        // 8-15
    int col    = (lane % 4) * 4;    // byte offset: 0,4,8,12
    uint4 a;
    a.x = *(const uint32_t*)(base + row_lo * stride + col);
    a.y = *(const uint32_t*)(base + row_hi * stride + col);
    a.z = *(const uint32_t*)(base + row_lo * stride + 16 + col);
    a.w = *(const uint32_t*)(base + row_hi * stride + 16 + col);
    return a;
}

// ---- Load MMA B-fragment from shared memory ----
// B is 8×64 INT4 in row-major packed format (8 weight rows × 32 bytes).
// For mma .col operand, B[k,n] = B_packed[n, k/2]:
//   Thread T: groupID = T/4 → selects which of the 8 weight rows (= output column)
//             localID = T%4 → which 8-element k-chunk
//   b.x = B[groupID, localID*4 .. localID*4+3]  (k=0..31)
//   b.y = B[groupID, 16+localID*4 .. 16+localID*4+3]  (k=32..63)
__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row  = lane / 4;          // weight row 0-7
    int col  = (lane % 4) * 4;
    uint2 b;
    b.x = *(const uint32_t*)(base + row * stride + col);
    b.y = *(const uint32_t*)(base + row * stride + 16 + col);
    return b;
}


// ---- Main GEMM kernel ----
__global__ void gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,
    const half    *__restrict__ scales_B,
    half          *__restrict__ C,
    int M, int N, int K, int group_size)
{
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int halfK = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    // Double-buffered shared memory
    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;
    const int tileB = BLOCK_N * SMEM_STRIDE;
    uint8_t *sA0 = smem, *sB0 = smem + tileA;
    uint8_t *sA1 = smem + tileA + tileB, *sB1 = sA1 + tileA;
    uint8_t *sA[2] = {sA0, sA1};
    uint8_t *sB[2] = {sB0, sB1};

    // FP32 accumulators: [n_tile][mma_half=0,1][4 values]
    // mma_half 0 = columns 0-7, mma_half 1 = columns 8-15
    float acc[TILES_N][2][4];
    for (int j = 0; j < TILES_N; j++)
        for (int h = 0; h < 2; h++)
            acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;

    // ---- Cooperative tile loader ----
    // 256 threads × 16 bytes = 4096 bytes per matrix. Thread tid loads one 16B chunk.
    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BLOCK_K / 2);  // byte offset in row
        // A: 128 rows × 32 bytes. tid → row=tid/2, half=tid%2
        {
            int row = tid / 2, half = tid % 2;
            bool p = (bm + row < M) && (kb + half * 16 < halfK);
            cp_async_16(sA[s] + row * SMEM_STRIDE + half * 16,
                        A + (size_t)(bm + row) * halfK + kb + half * 16, p);
        }
        // B: same
        {
            int row = tid / 2, half = tid % 2;
            bool p = (bn + row < N) && (kb + half * 16 < halfK);
            cp_async_16(sB[s] + row * SMEM_STRIDE + half * 16,
                        B + (size_t)(bn + row) * halfK + kb + half * 16, p);
        }
        cp_commit();
    };

    // Prefetch first tile
    if (num_k_tiles > 0) load_tile(0, 0);

    // ---- Main K-loop ----
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt & 1;
        if (kt + 1 < num_k_tiles) load_tile(kt + 1, (kt + 1) & 1);
        cp_wait(kt + 1 < num_k_tiles ? 1 : 0);
        __syncthreads();

        // Group index for scales
        int g = (kt * BLOCK_K) / group_size;

        // Activation scales for this warp's rows
        int m_lo = bm + warpId * WARP_M + laneId / 4;
        int m_hi = m_lo + 8;
        float sa_lo = (m_lo < M) ? __half2float(scales_A[m_lo * num_groups + g]) : 0.f;
        float sa_hi = (m_hi < M) ? __half2float(scales_A[m_hi * num_groups + g]) : 0.f;

        // Load A-fragment (one per warp, reused across N-tiles)
        uint4 af = load_a_frag(sA[s] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        // Process each 16-column N-tile
        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;

            // Two m16n8k64 MMAs per 16-col tile
            uint2 bf0 = load_b_frag(sB[s] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            uint2 bf1 = load_b_frag(sB[s] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            // Weight scales (per output column)
            // MMA output: c[0]=C[row_lo, col0], c[1]=C[row_lo, col1]
            //             c[2]=C[row_hi, col0], c[3]=C[row_hi, col1]
            // col0 = (laneId%4)*2, col1 = col0+1 (within the 8-col MMA tile)
            int c0 = bn + n_off + (laneId % 4) * 2;
            int c1 = c0 + 1;
            int c2 = c0 + 8;
            int c3 = c2 + 1;
            float sb0 = (c0 < N) ? __half2float(scales_B[c0 * num_groups + g]) : 0.f;
            float sb1 = (c1 < N) ? __half2float(scales_B[c1 * num_groups + g]) : 0.f;
            float sb2 = (c2 < N) ? __half2float(scales_B[c2 * num_groups + g]) : 0.f;
            float sb3 = (c3 < N) ? __half2float(scales_B[c3 * num_groups + g]) : 0.f;

            acc[nt][0][0] += (float)p0[0] * sa_lo * sb0;
            acc[nt][0][1] += (float)p0[1] * sa_lo * sb1;
            acc[nt][0][2] += (float)p0[2] * sa_hi * sb0;
            acc[nt][0][3] += (float)p0[3] * sa_hi * sb1;
            acc[nt][1][0] += (float)p1[0] * sa_lo * sb2;
            acc[nt][1][1] += (float)p1[1] * sa_lo * sb3;
            acc[nt][1][2] += (float)p1[2] * sa_hi * sb2;
            acc[nt][1][3] += (float)p1[3] * sa_hi * sb3;
        }
        __syncthreads();
    }

    // ---- Epilogue: write to global ----
    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;
    for (int nt = 0; nt < TILES_N; nt++) {
        int c0 = bn + nt * 16 + (laneId % 4) * 2;
        int c1 = c0 + 1, c2 = c0 + 8, c3 = c2 + 1;
        if (m_lo < M) {
            if (c0 < N) C[m_lo * N + c0] = __float2half(acc[nt][0][0]);
            if (c1 < N) C[m_lo * N + c1] = __float2half(acc[nt][0][1]);
            if (c2 < N) C[m_lo * N + c2] = __float2half(acc[nt][1][0]);
            if (c3 < N) C[m_lo * N + c3] = __float2half(acc[nt][1][1]);
        }
        if (m_hi < M) {
            if (c0 < N) C[m_hi * N + c0] = __float2half(acc[nt][0][2]);
            if (c1 < N) C[m_hi * N + c1] = __float2half(acc[nt][0][3]);
            if (c2 < N) C[m_hi * N + c2] = __float2half(acc[nt][1][2]);
            if (c3 < N) C[m_hi * N + c3] = __float2half(acc[nt][1][3]);
        }
    }
}


// ---- Host wrapper (same signature as naive -- drop-in replacement) ----
torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed, torch::Tensor B_packed,
    torch::Tensor scales_A, torch::Tensor scales_B, int group_size)
{
    TORCH_CHECK(A_packed.is_cuda() && B_packed.is_cuda());
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8);
    int M = A_packed.size(0), K = A_packed.size(1) * 2, N = B_packed.size(0);

    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARP_SZ * NUM_WARPS);
    int smem = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
