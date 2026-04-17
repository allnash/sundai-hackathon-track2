#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Naive INT4 GEMM kernel.
// Computes C[M, N] = A[M, K] @ B[N, K]^T where A and B are packed INT4 with per-group scales.
// Each thread computes one output element.
// Within each group, INT4 dot product is accumulated in int32, then scaled to FP32.
__global__ void gemm_int4_naive_kernel(
    const uint8_t* __restrict__ A,        // [M, K/2] packed INT4 activations
    const uint8_t* __restrict__ B,        // [N, K/2] packed INT4 weights
    const half* __restrict__ scales_A,    // [M, num_groups]
    const half* __restrict__ scales_B,    // [N, num_groups]
    half* __restrict__ C,                 // [M, N] output
    int M,
    int N,
    int K,
    int group_size
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    int num_groups = K / group_size;
    int half_group = group_size / 2;  // number of packed bytes per group

    float acc = 0.0f;

    for (int g = 0; g < num_groups; g++) {
        float sa = __half2float(scales_A[row * num_groups + g]);
        float sb = __half2float(scales_B[col * num_groups + g]);

        int dot = 0;
        int byte_base = g * half_group;

        for (int b = 0; b < half_group; b++) {
            uint8_t a_packed = A[row * (K / 2) + byte_base + b];
            uint8_t b_packed = B[col * (K / 2) + byte_base + b];

            // Unpack low nibble (even element) with sign extension
            int a_lo = (int)(a_packed & 0xF);
            if (a_lo >= 8) a_lo -= 16;
            int b_lo = (int)(b_packed & 0xF);
            if (b_lo >= 8) b_lo -= 16;

            // Unpack high nibble (odd element) with sign extension
            int a_hi = (int)((a_packed >> 4) & 0xF);
            if (a_hi >= 8) a_hi -= 16;
            int b_hi = (int)((b_packed >> 4) & 0xF);
            if (b_hi >= 8) b_hi -= 16;

            dot += a_lo * b_lo + a_hi * b_hi;
        }

        acc += sa * sb * (float)dot;
    }

    C[row * N + col] = __float2half(acc);
}

torch::Tensor gemm_int4_naive(
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

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_int4_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size
    );

    return C;
}
