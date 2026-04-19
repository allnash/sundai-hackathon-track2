/**
 * Optimized INT4 GEMM Kernel v4
 *
 * Key: SMEM_STRIDE=48 gives ZERO bank conflicts (proven by reference at 58 TOPs).
 * BLOCK_N=64 gives 4 blocks/SM = 32 warps (2x reference occupancy).
 */

#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <mutex>

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

// ====================== GEMM ======================

static constexpr int BM = 128;
static constexpr int BN = 64;       // Half of reference → 2x occupancy
static constexpr int BK = 64;       // = group_size, one group per K-tile
static constexpr int WS = 32;
static constexpr int NW = 8;
static constexpr int WM = BM / NW;  // 16
static constexpr int TN = BN / 16;  // 4
static constexpr int SS = BK / 2 + 16;  // 48 bytes — ZERO bank conflicts (proven)

// ---- MMA m16n8k64 ----
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

// ---- cp.async ----
__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.cg.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}
__device__ __forceinline__ void cp_commit()  { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_wait_0()  { asm volatile("cp.async.wait_group 0;\n"); }
__device__ __forceinline__ void cp_wait_1()  { asm volatile("cp.async.wait_group 1;\n"); }

// ---- Fragment loaders ----
__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WS;
    int rl = lane / 4, rh = rl + 8, c = (lane % 4) * 4;
    uint4 a;
    a.x = *(const uint32_t*)(base + rl * stride + c);
    a.y = *(const uint32_t*)(base + rh * stride + c);
    a.z = *(const uint32_t*)(base + rl * stride + 16 + c);
    a.w = *(const uint32_t*)(base + rh * stride + 16 + c);
    return a;
}

__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WS;
    int r = lane / 4, c = (lane % 4) * 4;
    uint2 b;
    b.x = *(const uint32_t*)(base + r * stride + c);
    b.y = *(const uint32_t*)(base + r * stride + 16 + c);
    return b;
}

// ---- GEMM kernel ----
__global__ void __launch_bounds__(256, 4)
gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half    *__restrict__ scales_A,
    const half    *__restrict__ scales_B,
    half          *__restrict__ C,
    int M, int N, int K, int group_size)
{
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int wid = tid / WS;
    const int lid = tid % WS;
    const int halfK = K / 2;
    const int ng = K / group_size;
    const int nkt = K / BK;

    // Double-buffered shared memory
    extern __shared__ uint8_t smem[];
    const int tA = BM * SS;   // 6144
    const int tB = BN * SS;   // 3072
    uint8_t *sA[2] = {smem, smem + tA + tB};
    uint8_t *sB[2] = {smem + tA, smem + tA + tB + tA};

    // Accumulators (32 float regs)
    float acc[TN][2][4];
    #pragma unroll
    for (int j = 0; j < TN; j++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            acc[j][h][0] = acc[j][h][1] = acc[j][h][2] = acc[j][h][3] = 0.f;

    // Tile loader
    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BK / 2);

        // A tile: 128 rows × 2 halves = 256 calls, all 256 threads
        {
            int row = tid / 2, half_idx = tid % 2;
            bool p = (bm + row < M) && (kb + half_idx * 16 < halfK);
            cp_async_16(sA[s] + row * SS + half_idx * 16,
                        A + (size_t)(bm + row) * halfK + kb + half_idx * 16, p);
        }

        // B tile: 64 rows × 2 halves = 128 calls, first 128 threads
        if (tid < BN * 2) {
            int row = tid / 2, half_idx = tid % 2;
            bool p = (bn + row < N) && (kb + half_idx * 16 < halfK);
            cp_async_16(sB[s] + row * SS + half_idx * 16,
                        B + (size_t)(bn + row) * halfK + kb + half_idx * 16, p);
        }

        cp_commit();
    };

    // Prefetch first tile
    if (nkt > 0) load_tile(0, 0);

    // Precompute row indices
    const int m_lo = bm + wid * WM + lid / 4;
    const int m_hi = m_lo + 8;

    // Main K-loop
    for (int kt = 0; kt < nkt; kt++) {
        int s = kt & 1;

        if (kt + 1 < nkt) load_tile(kt + 1, (kt + 1) & 1);
        if (kt + 1 < nkt) cp_wait_1(); else cp_wait_0();
        __syncthreads();

        int g = (kt * BK) / group_size;

        // A scales (L1 cached, 2 unique per warp)
        float sa_lo = (m_lo < M) ? __half2float(scales_A[m_lo * ng + g]) : 0.f;
        float sa_hi = (m_hi < M) ? __half2float(scales_A[m_hi * ng + g]) : 0.f;

        // A fragment (reused across N-tiles)
        uint4 af = load_a_frag(sA[s] + wid * WM * SS, SS);

        #pragma unroll
        for (int nt = 0; nt < TN; nt++) {
            int noff = nt * 16;

            uint2 bf0 = load_b_frag(sB[s] + noff * SS, SS);
            uint2 bf1 = load_b_frag(sB[s] + (noff + 8) * SS, SS);

            int p0[4] = {0,0,0,0}, p1[4] = {0,0,0,0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

            // B scales (L1 cached)
            int c0 = bn + noff + (lid % 4) * 2;
            int c1 = c0 + 1, c2 = c0 + 8, c3 = c2 + 1;
            float sb0 = (c0 < N) ? __half2float(scales_B[c0 * ng + g]) : 0.f;
            float sb1 = (c1 < N) ? __half2float(scales_B[c1 * ng + g]) : 0.f;
            float sb2 = (c2 < N) ? __half2float(scales_B[c2 * ng + g]) : 0.f;
            float sb3 = (c3 < N) ? __half2float(scales_B[c3 * ng + g]) : 0.f;

            // Precomputed scale products
            float s00 = sa_lo * sb0, s01 = sa_lo * sb1;
            float s10 = sa_hi * sb0, s11 = sa_hi * sb1;
            float s20 = sa_lo * sb2, s21 = sa_lo * sb3;
            float s30 = sa_hi * sb2, s31 = sa_hi * sb3;

            acc[nt][0][0] += (float)p0[0] * s00;
            acc[nt][0][1] += (float)p0[1] * s01;
            acc[nt][0][2] += (float)p0[2] * s10;
            acc[nt][0][3] += (float)p0[3] * s11;
            acc[nt][1][0] += (float)p1[0] * s20;
            acc[nt][1][1] += (float)p1[1] * s21;
            acc[nt][1][2] += (float)p1[2] * s30;
            acc[nt][1][3] += (float)p1[3] * s31;
        }
        __syncthreads();
    }

    // Epilogue
    #pragma unroll
    for (int nt = 0; nt < TN; nt++) {
        int c0 = bn + nt * 16 + (lid % 4) * 2;
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
    TORCH_CHECK(K % BK == 0, "K must be multiple of ", BK);
    TORCH_CHECK(group_size <= BK && BK % group_size == 0);

    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WS * NW);
    int smem = 2 * (BM * SS + BN * SS);  // 18432 bytes

    static std::once_flag flag;
    std::call_once(flag, [&]() {
        cudaFuncSetAttribute(gemm_int4_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(gemm_int4_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
    });

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(), B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size);
    return C;
}
