#include <cuda_fp16.h>
#include <cstdint>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

// Naive INT4 quantization kernel.
// Each thread handles one group of `group_size` elements in one row.
// Performs per-group symmetric quantization: scale = max(|x|) / 7
// Packs two signed INT4 values per byte: low nibble = even element, high nibble = odd element.
__global__ void quantize_int4_naive_kernel(
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

std::vector<torch::Tensor> quantize_int4_naive(torch::Tensor input, int group_size) {
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

    quantize_int4_naive_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<uint8_t>(),
        reinterpret_cast<half*>(scales.data_ptr<at::Half>()),
        M, K, group_size
    );

    return {output, scales};
}
