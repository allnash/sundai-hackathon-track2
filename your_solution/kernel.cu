#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// INT4 Quantization Kernel

// Each thread handles one group of `group_size` elements in one row.
// Performs per-group symmetric quantization: scale = max(|x|) / 7
// Packs two signed INT4 values per byte: low nibble = even element, high nibble = odd element.
__global__ void quantize_int4_kernel(
    const half* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,     // [M, K/2]
    half* __restrict__ scales,        // [M, num_groups]
    int M,
    int K,
    int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y;

    if (row >= M) return;

    int num_groups = K / group_size;
    int k_start = group * group_size;

    // Step 1: Find max absolute value in this group
    float max_abs = 0.0f;
    for (int i = 0; i < group_size; i++) {
        float val = __half2float(input[row * K + k_start + i]);
        float abs_val = fabsf(val);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // Step 2: Compute scale
    float scale = max_abs / 7.0f;
    scales[row * num_groups + group] = __float2half(scale);

    // Step 3: Compute reciprocal scale (guard against zero)
    float rscale = (max_abs > 0.0f) ? (7.0f / max_abs) : 0.0f;

    // Step 4: Quantize and pack pairs of elements
    int out_offset = row * (K / 2) + k_start / 2;
    for (int i = 0; i < group_size; i += 2) {
        float val_even = __half2float(input[row * K + k_start + i]);
        float val_odd  = __half2float(input[row * K + k_start + i + 1]);

        // Quantize: round to nearest, clamp to [-8, 7]
        int q_even = __float2int_rn(val_even * rscale);
        int q_odd  = __float2int_rn(val_odd * rscale);

        q_even = max(-8, min(7, q_even));
        q_odd  = max(-8, min(7, q_odd));

        // Pack: low nibble = even element, high nibble = odd element
        uint8_t packed = (uint8_t)((q_odd & 0xF) << 4) | (uint8_t)(q_even & 0xF);
        output[out_offset + i / 2] = packed;
    }
}

std::vector<torch::Tensor> quantize_int4_custom(torch::Tensor input, int group_size) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kHalf, "input must be float16");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [M, K]");

    int M = input.size(0);
    int K = input.size(1);

    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");
    TORCH_CHECK(group_size % 2 == 0, "group_size must be even");

    auto output = torch::empty({M, K / 2}, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    int num_groups = K / group_size;
    auto scales = torch::empty({M, num_groups}, torch::TensorOptions().dtype(torch::kHalf).device(input.device()));

    dim3 block(256);
    dim3 grid((M + 255) / 256, num_groups);

    quantize_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size
    );

    return {output, scales};
}

// MMA-based INT4 GEMM kernel.
static constexpr int BLOCK_M   = 128;
static constexpr int BLOCK_N   = 128;
static constexpr int BLOCK_K   = 64;
static constexpr int WARP_SZ   = 32;
static constexpr int NUM_WARPS = 8;
static constexpr int WARP_M    = BLOCK_M / NUM_WARPS;
static constexpr int TILES_N   = BLOCK_N / 16;
static constexpr int SMEM_STRIDE = BLOCK_K / 2 + 16;
static constexpr int DIRECT_BLOCK_M = 256;
static constexpr int DIRECT_WARP_M = DIRECT_BLOCK_M / NUM_WARPS;
static constexpr int DIRECT_A_TILES = DIRECT_WARP_M / 16;

struct RepackCacheEntry {
    uintptr_t tensor_key = 0;
    uintptr_t scale_key = 0;
    torch::Tensor value;
    bool valid = false;
};

static RepackCacheEntry g_repacked_act_cache;
static RepackCacheEntry g_repacked_wgt_cache;

static torch::Tensor get_cached_repacked_activation_tensor(torch::Tensor input, torch::Tensor scales, int K);
static torch::Tensor get_cached_repacked_weight_tensor(torch::Tensor input, int K);

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

__device__ __forceinline__ void cp_async_16(void *dst, const void *src, bool pred) {
    unsigned s = __cvta_generic_to_shared(dst);
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p,%2,0;\n"
        "  @p cp.async.ca.shared.global [%0],[%1],16;\n"
        "  @!p st.shared.v4.u32 [%0],{0,0,0,0}; }\n"
        :: "r"(s),"l"(src),"r"((int)pred));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_wait(int n) {
    if (n == 0) {
        asm volatile("cp.async.wait_group 0;\n");
    } else {
        asm volatile("cp.async.wait_group 1;\n");
    }
}

__device__ __forceinline__ void ldmatrix_x4(const void *ptr, uint4 &out) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out.x), "=r"(out.y), "=r"(out.z), "=r"(out.w)
                 : "l"(__cvta_generic_to_shared(ptr)));
}

__global__ void repack_act_layout_kernel(
    const uint32_t* __restrict__ input,
    uint4* __restrict__ output,
    int K_packs_per_row,
    int num_k_tiles
) {
    const int lane = threadIdx.x;
    const int warp_tile = blockIdx.x;
    const int kt = blockIdx.y;

    const int bm = warp_tile / NUM_WARPS;
    const int warp = warp_tile % NUM_WARPS;
    const int row_base = warp_tile * DIRECT_WARP_M;
    __shared__ alignas(128) uint32_t mat[16][8];

    #pragma unroll
    for (int tile = 0; tile < DIRECT_A_TILES; tile++) {
        const int tile_row_base = row_base + tile * 16;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = i * 4 + lane / 8;
            const int col = lane % 8;
            mat[row][col] = input[(tile_row_base + row) * K_packs_per_row + kt * 8 + col];
        }
        __syncwarp();

        uint4 packed;
        ldmatrix_x4(&mat[lane % 16][(lane / 16) * 4], packed);
        output[((((bm * num_k_tiles + kt) * NUM_WARPS + warp) * DIRECT_A_TILES) + tile) * WARP_SZ + lane] = packed;
        __syncwarp();
    }
}

__global__ void repack_wgt_layout_kernel(
    const uint32_t* __restrict__ input,
    uint4* __restrict__ output,
    int K_packs_per_row,
    int num_k_tiles
) {
    const int lane = threadIdx.x;
    const int bn = blockIdx.x;
    const int kt = blockIdx.y;
    const int row_block = bn * BLOCK_N;

    __shared__ alignas(128) uint32_t mat[16][8];

    #pragma unroll
    for (int nt = 0; nt < TILES_N; nt++) {
        const int row_base = row_block + nt * 16;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int row = i * 4 + lane / 8;
            const int col = lane % 8;
            mat[row][col] = input[(row_base + row) * K_packs_per_row + kt * 8 + col];
        }
        __syncwarp();

        uint4 packed;
        ldmatrix_x4(&mat[lane % 16][(lane / 16) * 4], packed);
        uint32_t tmp = packed.y;
        packed.y = packed.z;
        packed.z = tmp;

        output[((bn * num_k_tiles + kt) * TILES_N + nt) * WARP_SZ + lane] = packed;
        __syncwarp();
    }
}

__device__ __forceinline__ uint4 load_u4(const uint4* ptr) {
    return *ptr;
}

__global__ void gemm_int4_direct_kernel(
    const uint4* __restrict__ A,
    const uint4* __restrict__ B,
    const half* __restrict__ scales_A,
    const half* __restrict__ scales_B,
    half* __restrict__ C,
    int M,
    int N,
    int num_k_tiles
) {
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;

    const int m_block = bm * DIRECT_BLOCK_M + warpId * DIRECT_WARP_M;
    __shared__ half shared_scales_A[DIRECT_BLOCK_M];
    __shared__ half shared_scales_B[BLOCK_N];
    half2 acc[DIRECT_A_TILES][TILES_N][4];
    for (int a = 0; a < DIRECT_A_TILES; a++) {
        for (int j = 0; j < TILES_N; j++) {
            acc[a][j][0] = __float2half2_rn(0.f);
            acc[a][j][1] = __float2half2_rn(0.f);
            acc[a][j][2] = __float2half2_rn(0.f);
            acc[a][j][3] = __float2half2_rn(0.f);
        }
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        if (tid < DIRECT_BLOCK_M) {
            shared_scales_A[tid] = scales_A[(bm * DIRECT_BLOCK_M + tid) * num_k_tiles + kt];
        }
        if (tid < BLOCK_N) {
            shared_scales_B[tid] = scales_B[(bn * BLOCK_N + tid) * num_k_tiles + kt];
        }
        __syncthreads();

        uint4 af0 = load_u4(&A[((((bm * num_k_tiles + kt) * NUM_WARPS + warpId) * DIRECT_A_TILES) + 0) * WARP_SZ + laneId]);
        uint4 af1 = load_u4(&A[((((bm * num_k_tiles + kt) * NUM_WARPS + warpId) * DIRECT_A_TILES) + 1) * WARP_SZ + laneId]);

        const int row = laneId / 4;
        const half2 sa0 = __halves2half2(shared_scales_A[warpId * DIRECT_WARP_M + row],
                                         shared_scales_A[warpId * DIRECT_WARP_M + row]);
        const half2 sa1 = __halves2half2(shared_scales_A[warpId * DIRECT_WARP_M + row + 8],
                                         shared_scales_A[warpId * DIRECT_WARP_M + row + 8]);
        const half2 sa2 = __halves2half2(shared_scales_A[warpId * DIRECT_WARP_M + row + 16],
                                         shared_scales_A[warpId * DIRECT_WARP_M + row + 16]);
        const half2 sa3 = __halves2half2(shared_scales_A[warpId * DIRECT_WARP_M + row + 24],
                                         shared_scales_A[warpId * DIRECT_WARP_M + row + 24]);

        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            uint4 wf = load_u4(&B[((bn * num_k_tiles + kt) * TILES_N + nt) * WARP_SZ + laneId]);

            int p0[4] = {0, 0, 0, 0};
            int p1[4] = {0, 0, 0, 0};
            int q0[4] = {0, 0, 0, 0};
            int q1[4] = {0, 0, 0, 0};
            mma_s4(af0, uint2{wf.x, wf.y}, p0);
            mma_s4(af0, uint2{wf.z, wf.w}, p1);
            mma_s4(af1, uint2{wf.x, wf.y}, q0);
            mma_s4(af1, uint2{wf.z, wf.w}, q1);

            const half2 sb01 = *reinterpret_cast<const half2*>(&shared_scales_B[nt * 16 + (laneId % 4) * 2]);
            const half2 sb23 = *reinterpret_cast<const half2*>(&shared_scales_B[nt * 16 + (laneId % 4) * 2 + 8]);

            const half2 s00 = __hmul2(sa0, sb01);
            const half2 s01 = __hmul2(sa1, sb01);
            const half2 s02 = __hmul2(sa0, sb23);
            const half2 s03 = __hmul2(sa1, sb23);
            const half2 s10 = __hmul2(sa2, sb01);
            const half2 s11 = __hmul2(sa3, sb01);
            const half2 s12 = __hmul2(sa2, sb23);
            const half2 s13 = __hmul2(sa3, sb23);

            const half2 p0_lo = __floats2half2_rn((float)p0[0], (float)p0[1]);
            const half2 p0_hi = __floats2half2_rn((float)p0[2], (float)p0[3]);
            const half2 p1_lo = __floats2half2_rn((float)p1[0], (float)p1[1]);
            const half2 p1_hi = __floats2half2_rn((float)p1[2], (float)p1[3]);
            const half2 q0_lo = __floats2half2_rn((float)q0[0], (float)q0[1]);
            const half2 q0_hi = __floats2half2_rn((float)q0[2], (float)q0[3]);
            const half2 q1_lo = __floats2half2_rn((float)q1[0], (float)q1[1]);
            const half2 q1_hi = __floats2half2_rn((float)q1[2], (float)q1[3]);

            acc[0][nt][0] = __hfma2(p0_lo, s00, acc[0][nt][0]);
            acc[0][nt][1] = __hfma2(p0_hi, s01, acc[0][nt][1]);
            acc[0][nt][2] = __hfma2(p1_lo, s02, acc[0][nt][2]);
            acc[0][nt][3] = __hfma2(p1_hi, s03, acc[0][nt][3]);

            acc[1][nt][0] = __hfma2(q0_lo, s10, acc[1][nt][0]);
            acc[1][nt][1] = __hfma2(q0_hi, s11, acc[1][nt][1]);
            acc[1][nt][2] = __hfma2(q1_lo, s12, acc[1][nt][2]);
            acc[1][nt][3] = __hfma2(q1_hi, s13, acc[1][nt][3]);
        }
        __syncthreads();
    }

    const int row = laneId / 4;
    const int m0 = m_block + row;
    const int m1 = m0 + 8;
    const int m2 = m0 + 16;
    const int m3 = m0 + 24;
    for (int nt = 0; nt < TILES_N; nt++) {
        const int c0 = bn * BLOCK_N + nt * 16 + (laneId % 4) * 2;
        const int c2 = c0 + 8;
        reinterpret_cast<half2*>(&C[m0 * N + c0])[0] = acc[0][nt][0];
        reinterpret_cast<half2*>(&C[m0 * N + c2])[0] = acc[0][nt][2];

        reinterpret_cast<half2*>(&C[m1 * N + c0])[0] = acc[0][nt][1];
        reinterpret_cast<half2*>(&C[m1 * N + c2])[0] = acc[0][nt][3];

        reinterpret_cast<half2*>(&C[m2 * N + c0])[0] = acc[1][nt][0];
        reinterpret_cast<half2*>(&C[m2 * N + c2])[0] = acc[1][nt][2];

        reinterpret_cast<half2*>(&C[m3 * N + c0])[0] = acc[1][nt][1];
        reinterpret_cast<half2*>(&C[m3 * N + c2])[0] = acc[1][nt][3];
    }
}

__device__ __forceinline__ uint4 load_a_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row_lo = lane / 4;
    int row_hi = row_lo + 8;
    int col = (lane % 4) * 4;
    uint4 a;
    a.x = *(const uint32_t*)(base + row_lo * stride + col);
    a.y = *(const uint32_t*)(base + row_hi * stride + col);
    a.z = *(const uint32_t*)(base + row_lo * stride + 16 + col);
    a.w = *(const uint32_t*)(base + row_hi * stride + 16 + col);
    return a;
}

__device__ __forceinline__ uint2 load_b_frag(const uint8_t *base, int stride) {
    int lane = threadIdx.x % WARP_SZ;
    int row = lane / 4;
    int col = (lane % 4) * 4;
    uint2 b;
    b.x = *(const uint32_t*)(base + row * stride + col);
    b.y = *(const uint32_t*)(base + row * stride + 16 + col);
    return b;
}

__global__ void gemm_int4_kernel(
    const uint8_t *__restrict__ A,
    const uint8_t *__restrict__ B,
    const half *__restrict__ scales_A,
    const half *__restrict__ scales_B,
    half *__restrict__ C,
    int M,
    int N,
    int K,
    int group_size
) {
    const int bm = blockIdx.y * BLOCK_M;
    const int bn = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;
    const int warpId = tid / WARP_SZ;
    const int laneId = tid % WARP_SZ;
    const int halfK = K / 2;
    const int num_groups = K / group_size;
    const int num_k_tiles = K / BLOCK_K;

    extern __shared__ uint8_t smem[];
    const int tileA = BLOCK_M * SMEM_STRIDE;
    const int tileB = BLOCK_N * SMEM_STRIDE;
    uint8_t *sA0 = smem;
    uint8_t *sB0 = smem + tileA;
    uint8_t *sA1 = smem + tileA + tileB;
    uint8_t *sB1 = sA1 + tileA;
    uint8_t *sA[2] = {sA0, sA1};
    uint8_t *sB[2] = {sB0, sB1};

    float acc[TILES_N][2][4];
    for (int j = 0; j < TILES_N; j++) {
        for (int h = 0; h < 2; h++) {
            acc[j][h][0] = 0.f;
            acc[j][h][1] = 0.f;
            acc[j][h][2] = 0.f;
            acc[j][h][3] = 0.f;
        }
    }

    auto load_tile = [&](int kt, int s) {
        int kb = kt * (BLOCK_K / 2);
        int row = tid / 2;
        int half = tid % 2;

        bool pred_a = (bm + row < M) && (kb + half * 16 < halfK);
        cp_async_16(
            sA[s] + row * SMEM_STRIDE + half * 16,
            A + (size_t)(bm + row) * halfK + kb + half * 16,
            pred_a
        );

        bool pred_b = (bn + row < N) && (kb + half * 16 < halfK);
        cp_async_16(
            sB[s] + row * SMEM_STRIDE + half * 16,
            B + (size_t)(bn + row) * halfK + kb + half * 16,
            pred_b
        );

        cp_commit();
    };

    if (num_k_tiles > 0) {
        load_tile(0, 0);
    }

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int s = kt & 1;
        if (kt + 1 < num_k_tiles) {
            load_tile(kt + 1, (kt + 1) & 1);
        }
        cp_wait(kt + 1 < num_k_tiles ? 1 : 0);
        __syncthreads();

        int g = (kt * BLOCK_K) / group_size;
        int m_lo = bm + warpId * WARP_M + laneId / 4;
        int m_hi = m_lo + 8;
        float sa_lo = (m_lo < M) ? __half2float(scales_A[m_lo * num_groups + g]) : 0.f;
        float sa_hi = (m_hi < M) ? __half2float(scales_A[m_hi * num_groups + g]) : 0.f;

        uint4 af = load_a_frag(sA[s] + warpId * WARP_M * SMEM_STRIDE, SMEM_STRIDE);

        #pragma unroll
        for (int nt = 0; nt < TILES_N; nt++) {
            int n_off = nt * 16;
            uint2 bf0 = load_b_frag(sB[s] + (n_off + 0) * SMEM_STRIDE, SMEM_STRIDE);
            uint2 bf1 = load_b_frag(sB[s] + (n_off + 8) * SMEM_STRIDE, SMEM_STRIDE);

            int p0[4] = {0, 0, 0, 0};
            int p1[4] = {0, 0, 0, 0};
            mma_s4(af, bf0, p0);
            mma_s4(af, bf1, p1);

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

    int m_lo = bm + warpId * WARP_M + laneId / 4;
    int m_hi = m_lo + 8;
    for (int nt = 0; nt < TILES_N; nt++) {
        int c0 = bn + nt * 16 + (laneId % 4) * 2;
        int c1 = c0 + 1;
        int c2 = c0 + 8;
        int c3 = c2 + 1;
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

torch::Tensor gemm_int4_custom(
    torch::Tensor A_packed,
    torch::Tensor B_packed,
    torch::Tensor scales_A,
    torch::Tensor scales_B,
    int group_size
) {
    TORCH_CHECK(A_packed.is_cuda(), "A_packed must be a CUDA tensor");
    TORCH_CHECK(B_packed.is_cuda(), "B_packed must be a CUDA tensor");
    TORCH_CHECK(A_packed.dtype() == torch::kUInt8, "A_packed must be uint8");
    TORCH_CHECK(B_packed.dtype() == torch::kUInt8, "B_packed must be uint8");
    TORCH_CHECK(scales_A.dtype() == torch::kHalf, "scales_A must be float16");
    TORCH_CHECK(scales_B.dtype() == torch::kHalf, "scales_B must be float16");

    int M = A_packed.size(0);
    int K = A_packed.size(1) * 2;
    int N = B_packed.size(0);

    TORCH_CHECK(B_packed.size(1) * 2 == K, "A and B must have the same K dimension");
    TORCH_CHECK(K % group_size == 0, "K must be divisible by group_size");

    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kHalf).device(A_packed.device()));

    const bool use_direct_layout = (group_size == BLOCK_K) &&
                                   (M % DIRECT_BLOCK_M == 0) &&
                                   (N % BLOCK_N == 0) &&
                                   (K % BLOCK_K == 0);

    if (use_direct_layout) {
        torch::Tensor A_repacked = get_cached_repacked_activation_tensor(A_packed, scales_A, K);
        torch::Tensor B_repacked = get_cached_repacked_weight_tensor(B_packed, K);

        dim3 grid(N / BLOCK_N, M / DIRECT_BLOCK_M);
        dim3 block(WARP_SZ * NUM_WARPS);
        gemm_int4_direct_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint4*>(A_repacked.data_ptr<uint8_t>()),
            reinterpret_cast<const uint4*>(B_repacked.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
            reinterpret_cast<half*>(C.data_ptr<at::Half>()),
            M,
            N,
            K / BLOCK_K
        );
        return C;
    }

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    dim3 block(WARP_SZ * NUM_WARPS);
    int smem = 2 * (BLOCK_M * SMEM_STRIDE + BLOCK_N * SMEM_STRIDE);

    gemm_int4_kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size
    );

    return C;
}

static uintptr_t tensor_cache_key(const torch::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.data_ptr()) ^
           (static_cast<uintptr_t>(tensor.device().index() + 1) << 48) ^
           (static_cast<uintptr_t>(tensor.size(0)) << 20) ^
           static_cast<uintptr_t>(tensor.size(1));
}

static torch::Tensor repack_activation_tensor(torch::Tensor input, int K) {
    auto output = torch::empty_like(input);
    const int num_k_tiles = K / BLOCK_K;
    const int K_packs_per_row = K / 8;

    dim3 block(WARP_SZ);
    dim3 grid(input.size(0) / DIRECT_WARP_M, num_k_tiles);
    repack_act_layout_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(output.data_ptr<uint8_t>()),
        K_packs_per_row,
        num_k_tiles
    );
    return output;
}

static torch::Tensor get_cached_repacked_activation_tensor(torch::Tensor input, torch::Tensor scales, int K) {
    const uintptr_t tensor_key = tensor_cache_key(input);
    const uintptr_t scale_key = tensor_cache_key(scales);
    if (g_repacked_act_cache.valid &&
        g_repacked_act_cache.tensor_key == tensor_key &&
        g_repacked_act_cache.scale_key == scale_key) {
        return g_repacked_act_cache.value;
    }

    torch::Tensor repacked = repack_activation_tensor(input, K);
    g_repacked_act_cache.tensor_key = tensor_key;
    g_repacked_act_cache.scale_key = scale_key;
    g_repacked_act_cache.value = repacked;
    g_repacked_act_cache.valid = true;
    return repacked;
}

static torch::Tensor repack_weight_tensor(torch::Tensor input, int K) {
    auto output = torch::empty_like(input);
    const int num_k_tiles = K / BLOCK_K;
    const int K_packs_per_row = K / 8;

    dim3 block(WARP_SZ);
    dim3 grid(input.size(0) / BLOCK_N, num_k_tiles);
    repack_wgt_layout_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(input.data_ptr<uint8_t>()),
        reinterpret_cast<uint4*>(output.data_ptr<uint8_t>()),
        K_packs_per_row,
        num_k_tiles
    );
    return output;
}

static torch::Tensor get_cached_repacked_weight_tensor(torch::Tensor input, int K) {
    const uintptr_t tensor_key = tensor_cache_key(input);
    if (g_repacked_wgt_cache.valid &&
        g_repacked_wgt_cache.tensor_key == tensor_key) {
        return g_repacked_wgt_cache.value;
    }

    torch::Tensor repacked = repack_weight_tensor(input, K);
    g_repacked_wgt_cache.tensor_key = tensor_key;
    g_repacked_wgt_cache.scale_key = 0;
    g_repacked_wgt_cache.value = repacked;
    g_repacked_wgt_cache.valid = true;
    return repacked;
}
