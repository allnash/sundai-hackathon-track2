/**
 * Optimized INT4 GEMM Kernel
 *
 * Key optimizations over reference MMA kernel:
 *   - Scales loaded cooperatively into shared memory (eliminates ~34 global loads per warp per K step)
 *   - Double-buffered scale tiles alongside data tiles
 *   - __launch_bounds__ for optimal register allocation
 *   - Max shared memory carveout for higher occupancy
 */

#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// ====================== QUANTIZATION KERNEL ======================

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

// ====================== OPTIMIZED INT4 GEMM ======================

static constexpr int BLOCK_M   = 128;
static constexpr int BLOCK_N   = 128;
static constexpr int BLOCK_K   = 64;
static constexpr int WARP_SZ   = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M    = BLOCK_M / NUM_WARPS;  // 16
static constexpr int TILES_N   = BLOCK_N / 16;         // 8
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;   // 48 bytes (16-byte aligned rows)

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

// ---- cp.async helpers ----
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

// ---- Load MMA A-fragment from shared memory (16×64 INT4, row-major) ----
__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4, row_hi = row_lo + 8;
    int col = (lane % 4) * 4;
    uint4 a;
    a.x = *(const uint32_t*)(base + row_lo * stride + col);
    a.y = *(const uint32_t*)(base + row_hi * stride + col);
    a.z = *(const uint32_t*)(base + row_lo * stride + 16 + col);
    a.w = *(const uint32_t*)(base + row_hi * stride + 16 + col);
    return a;
}

// ---- Load MMA B-fragment from shared memory (8×64 INT4, row-major) ----
__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row = lane / 4, col = (lane % 4) * 4;
    uint2 b;
    b.x = *(const uint32_t*)(base + row * stride + col);
    b.y = *(const uint32_t*)(base + row * stride + 16 + col);
    return b;
}

// ---- Main GEMM kernel ----
__global__ void __launch_bounds__(256, 2)
gemm_int4_kernel(
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

    // ---- Shared memory layout ----
    // [A tile 0][B tile 0][A tile 1][B tile 1][Scales A0][Scales B0][Scales A1][Scales B1]
    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;  // 6144
    const int tileB = BLOCK_N * SMEM_STRIDE;  // 6144
    uint8_t *sA[2] = {smem, smem + tileA + tileB};
    uint8_t *sB[2] = {smem + tileA, smem + tileA + tileB + tileA};

    // Double-buffered scales in shared memory (after data tiles)
    const int data_total = 2 * (tileA + tileB);  // 24576
    half *sScA[2] = {
        (half*)(smem + data_total),
        (half*)(smem + data_total + (BLOCK_M + BLOCK_N) * sizeof(half))
    };
    half *sScB[2] = {
        (half*)(smem + data_total + BLOCK_M * sizeof(half)),
        (half*)(smem + data_total + (BLOCK_M + BLOCK_N) * sizeof(half) + BLOCK_M * sizeof(half))
    };

    // ---- Accumulators ----
    float acc[TILES_N][2][4];
    #pragma unroll
    for (int j = 0; j < TILES_N; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;

    // ---- Cooperative tile + scale loader ----
    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BLOCK_K / 2);
        int g = (kt * BLOCK_K) / group_size;

        // Load A tile (128 rows × 32 bytes) via cp.async
        {
            int row = tid / 2, half_idx = tid % 2;
            bool p = (bm + row < M) && (kb + half_idx * 16 < halfK);
            cp_async_16(sA[s] + row * SMEM_STRIDE + half_idx * 16,
                        A + (size_t)(bm + row) * halfK + kb + half_idx * 16, p);
        }
        // Load B tile (128 rows × 32 bytes) via cp.async
        {
            int row = tid / 2, half_idx = tid % 2;
            bool p = (bn + row < N) && (kb + half_idx * 16 < halfK);
            cp_async_16(sB[s] + row * SMEM_STRIDE + half_idx * 16,
                        B + (size_t)(bn + row) * halfK + kb + half_idx * 16, p);
        }

        // Load scales cooperatively (256 threads → 128 A scales + 128 B scales)
        if (tid < BLOCK_M) {
            int m = bm + tid;
            sScA[s][tid] = (m < M) ? scales_A[m * num_groups + g] : __float2half(0.f);
        }
        if (tid >= BLOCK_M && tid < BLOCK_M + BLOCK_N) {
            int idx = tid - BLOCK_M;
            int n = bn + idx;
            sScB[s][idx] = (n < N) ? scales_B[n * num_groups + g] : __float2half(0.f);
        }

        cp_commit();
    };

    // Prefetch first tile
    if (num_k_tiles > 0) load_tile(0, 0);

    // ---- Main K-loop ----
    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt & 1;

        // Prefetch next tile (data + scales) to the other buffer
        if (kt + 1 < num_k_tiles) load_tile(kt + 1, (kt + 1) & 1);

        // Wait for current tile's async copies
        cp_wait(kt + 1 < num_k_tiles ? 1 : 0);
        __syncthreads();

        // ---- Read activation scales from shared memory ----
        float sa_lo = __half2float(sScA[s][warpId * WARP_M + laneId / 4]);
        float sa_hi = __half2float(sScA[s][warpId * WARP_M + laneId / 4 + 8]);

        // ---- Load A-fragment (reused across all N-tiles) ----
        uint4 af = load_a_frag(sA[s] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        // ---- Process each 16-column N-tile ----
        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;

            // Two m16n8k64 MMAs per 16-column tile
            uint2 bf0 = load_b_frag(sB[s] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            uint2 bf1 = load_b_frag(sB[s] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            // ---- Weight scales from shared memory (not global!) ----
            int c_off = n_off + (laneId % 4) * 2;
            float sb0 = __half2float(sScB[s][c_off]);
            float sb1 = __half2float(sScB[s][c_off + 1]);
            float sb2 = __half2float(sScB[s][c_off + 8]);
            float sb3 = __half2float(sScB[s][c_off + 9]);

            // Scale and accumulate
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

    // ---- Epilogue: write to global memory ----
    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;
    #pragma unroll
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

// ---- Host wrapper ----
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
    dim3 block(WARP_SZ * NUM_WARPS);  // 256

    // Shared memory: double-buffered data tiles + double-buffered scale arrays
    int smem_data   = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);  // 24576
    int smem_scales = 2 * (BLOCK_M + BLOCK_N) * sizeof(half);                // 1024
    int smem = smem_data + smem_scales;                                       // 25600

    // Configure for max shared memory to allow 2 blocks per SM
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(gemm_int4_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(gemm_int4_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        configured = true;
    }

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
