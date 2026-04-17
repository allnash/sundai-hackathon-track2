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

// INT4 GEMM Kernel

// Computes C[M, N] = A[M, K] @ B[N, K]^T where A and B are packed INT4 with per-group scales.
// Each thread computes one output element.
// Within each group, INT4 dot product is accumulated in int32, then scaled to FP32.
__global__ void gemm_int4_kernel(
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

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    gemm_int4_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        A_packed.data_ptr<uint8_t>(),
        B_packed.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scales_A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales_B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K, group_size
    );

    return C;
}
