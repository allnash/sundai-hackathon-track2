"""CUDA Challenge: INT4 Quantize + GEMM Benchmark

Loads real (activation, weight) pairs from FLUX.1-schnell, runs offline weight
quantization via the participant's quantize.py, then benchmarks the online
activation quantization and GEMM kernels.

Target shapes (M x N x K):
    4096 x  9216 x  3072  (attn_to_qkv)
    4096 x  3072 x  3072  (attn_to_out)
    4096 x 12288 x  3072  (ff_up)
    4096 x  3072 x 12288  (ff_down)

Usage:
    python benchmark.py
    python benchmark.py --group-size 128
"""

import os
import sys
import importlib.util
from pathlib import Path

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GROUP_SIZE = 64

# Per-layer cosine thresholds (set between g=128 and g=64 accuracy).
# Measured with nunchaku quantization on real FLUX.1-schnell weights.
COSINE_THRESHOLDS = {
    "attn_to_qkv": 0.989,   # g128=0.9870, g64=0.9910
    "attn_to_out":  0.991,   # g128=0.9905, g64=0.9921
    "ff_up":        0.978,   # g128=0.9745, g64=0.9819
    "ff_down":      0.977,   # g128=0.9714, g64=0.9822
}
COSINE_THRESHOLD_DEFAULT = 0.98


# ---- CUDA build ----

def get_cuda_flags():
    """Build CUDA compiler flags with auto-detected GPU architecture."""
    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}"
    code = f"sm_{cap[0]}{cap[1]}"
    return [
        "-O2",
        f"-gencode=arch={arch},code={code}",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
    ]


def build_modules():
    """JIT-compile reference and solution CUDA modules."""
    from torch.utils.cpp_extension import load

    cuda_flags = get_cuda_flags()
    cpp_flags = ["-O2"]

    print("Building reference kernels...")
    ref = load(
        name="ref_kernels",
        sources=[
            os.path.join(SCRIPT_DIR, "reference", "pybind.cpp"),
            os.path.join(SCRIPT_DIR, "reference", "quantize_int4.cu"),
            os.path.join(SCRIPT_DIR, "reference", "gemm_int4.cu"),
        ],
        extra_cflags=cpp_flags,
        extra_cuda_cflags=cuda_flags,
        verbose=False,
    )

    print("Building solution kernels...")
    try:
        sol = load(
            name="sol_kernels",
            sources=[
                os.path.join(SCRIPT_DIR, "your_solution", "pybind.cpp"),
                os.path.join(SCRIPT_DIR, "your_solution", "kernel.cu"),
            ],
            extra_cflags=cpp_flags,
            extra_cuda_cflags=cuda_flags,
            verbose=False,
        )
    except Exception as e:
        print(f"\nERROR: Your solution failed to compile:\n{e}")
        sys.exit(1)

    return ref, sol


# ---- Data loading ----

def load_quantize_module(path):
    """Dynamically load a quantize.py module."""
    spec = importlib.util.spec_from_file_location("quantize_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_flux_data(data_dir):
    """Load weights and activations from flux_dump/.

    Returns:
        weights: dict {layer_name: tensor [N, K] fp16}
        activations: dict {layer_name: tensor [M, K] fp16}
    """
    data_dir = Path(data_dir)

    weights_path = data_dir / "weights.pt"
    if not weights_path.exists():
        print(f"ERROR: {weights_path} not found. Run 'python dump_data.py' first.")
        sys.exit(1)

    weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Load activations (single resolution: 1024x1024)
    act_path = data_dir / "activations_1024x1024.pt"
    if not act_path.exists():
        print(f"ERROR: {act_path} not found. Run 'python dump_data.py' first.")
        sys.exit(1)

    activations = torch.load(act_path, map_location="cpu", weights_only=True)

    return weights, activations


# ---- Math helpers ----

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return (torch.dot(a_flat, b_flat) / (a_flat.norm() * b_flat.norm())).item()


def compute_quantize_bytes(M, K, group_size):
    """Total bytes moved for quantization: input + packed output + scales."""
    input_bytes = M * K * 2              # FP16 input
    packed_bytes = M * (K // 2)          # uint8 packed output
    scale_bytes = M * (K // group_size) * 2  # FP16 scales
    return input_bytes + packed_bytes + scale_bytes


def compute_gemm_bytes(M, N, K, group_size):
    """Total bytes moved for GEMM: packed A + packed B + scales + output."""
    a_bytes = M * (K // 2)
    b_bytes = N * (K // 2)
    sa_bytes = M * (K // group_size) * 2
    sb_bytes = N * (K // group_size) * 2
    c_bytes = M * N * 2
    return a_bytes + b_bytes + sa_bytes + sb_bytes + c_bytes


def compute_gemm_ops(M, N, K):
    """Total operations for GEMM: 2 * M * N * K (one multiply + one add per MAC)."""
    return 2 * M * N * K


# ---- Benchmarking ----

def benchmark_kernel(fn, args, warmup=5, iters=20):
    """Time a CUDA kernel using CUDA events. Returns median time in seconds."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    times.sort()
    return times[len(times) // 2]


# ---- Correctness ----

def check_correctness(ref, sol, activation, weight, wgt_packed, wgt_scales, group_size,
                      threshold=None):
    """Check solution vs FP16 matmul reference.

    Returns (passed, cosine, ref_cosine).
    """
    if threshold is None:
        threshold = COSINE_THRESHOLD_DEFAULT
    activation = activation.cuda().contiguous()
    weight = weight.cuda().contiguous()
    wgt_packed = wgt_packed.cuda().contiguous()
    wgt_scales = wgt_scales.cuda().contiguous()

    # Ground truth: FP16 matmul
    C_ref = (activation.float() @ weight.float().T).half()

    # Sanity: reference CUDA kernels
    ref_act_p, ref_act_s = ref.quantize_int4_naive(activation, group_size)
    ref_wgt_p, ref_wgt_s = ref.quantize_int4_naive(weight, group_size)
    C_ref_cuda = ref.gemm_int4_naive(ref_act_p, ref_wgt_p, ref_act_s, ref_wgt_s, group_size)
    ref_cos = cosine_similarity(C_ref_cuda, C_ref)

    # Solution: online activation quantize + GEMM with offline-quantized weights
    sol_act_p, sol_act_s = sol.quantize_int4(activation, group_size)
    C_sol = sol.gemm_int4(sol_act_p, wgt_packed, sol_act_s, wgt_scales, group_size)

    cos = cosine_similarity(C_sol, C_ref)
    passed = cos > threshold
    return passed, cos, ref_cos


# ---- Benchmark runner ----

def run_benchmark(ref, sol, activation, weight, wgt_packed, wgt_scales, group_size,
                  warmup=5, iters=20):
    """Benchmark activation quantize + GEMM. Returns (quant_gbs, gemm_tops, ref_gemm_tops)."""
    activation = activation.cuda().contiguous()
    weight = weight.cuda().contiguous()
    wgt_packed = wgt_packed.cuda().contiguous()
    wgt_scales = wgt_scales.cuda().contiguous()

    M, K = activation.shape
    N = wgt_packed.shape[0]

    # Benchmark activation quantize (solution)
    quant_time = benchmark_kernel(
        lambda a, gs: sol.quantize_int4(a, gs),
        [activation, group_size],
        warmup=warmup, iters=iters,
    )
    quant_bytes = compute_quantize_bytes(M, K, group_size)
    quant_gbs = quant_bytes / quant_time / 1e9

    # Prepare quantized activation for GEMM
    sol_act_p, sol_act_s = sol.quantize_int4(activation, group_size)

    # Benchmark GEMM (solution)
    gemm_time = benchmark_kernel(
        lambda ap, wp, as_, ws, gs: sol.gemm_int4(ap, wp, as_, ws, gs),
        [sol_act_p, wgt_packed, sol_act_s, wgt_scales, group_size],
        warmup=warmup, iters=iters,
    )
    gemm_ops = compute_gemm_ops(M, N, K)
    gemm_tops = gemm_ops / gemm_time / 1e12

    # Benchmark GEMM (reference) for speedup calculation
    ref_act_p, ref_act_s = ref.quantize_int4_naive(activation, group_size)
    ref_wgt_p, ref_wgt_s = ref.quantize_int4_naive(weight, group_size)
    ref_gemm_time = benchmark_kernel(
        lambda ap, wp, as_, ws, gs: ref.gemm_int4_naive(ap, wp, as_, ws, gs),
        [ref_act_p, ref_wgt_p, ref_act_s, ref_wgt_s, group_size],
        warmup=warmup, iters=iters,
    )
    ref_gemm_tops = gemm_ops / ref_gemm_time / 1e12

    return quant_gbs, gemm_tops, ref_gemm_tops


# ---- Main ----

# Presentation order for layers
LAYER_ORDER = ["attn_to_qkv", "attn_to_out", "ff_up", "ff_down"]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="INT4 Quantize + GEMM Benchmark")
    parser.add_argument("--data-dir", type=str, default=os.path.join(SCRIPT_DIR, "flux_dump"))
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    gs = args.group_size

    # Build CUDA modules
    ref, sol = build_modules()

    # Load flux data
    print("\nLoading flux data...")
    weights, activations = load_flux_data(args.data_dir)

    layer_names = [n for n in LAYER_ORDER if n in weights]
    print(f"  Layers: {layer_names}")

    # Load participant's quantize.py
    quantize_path = os.path.join(SCRIPT_DIR, "your_solution", "quantize.py")
    if not os.path.exists(quantize_path):
        print(f"ERROR: {quantize_path} not found.")
        sys.exit(1)
    quantize_mod = load_quantize_module(quantize_path)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {gpu_name}")
    print(f"Group size: {gs}")
    print(f"Cosine thresholds: {COSINE_THRESHOLDS}")

    # ---- Offline weight quantization (NOT timed) ----
    print("\nRunning offline weight quantization...")
    quantized_weights = {}
    for name in layer_names:
        w = weights[name]
        result = quantize_mod.quantize_weights(w, group_size=gs)
        quantized_weights[name] = result
        N, K = w.shape
        print(f"  {name}: [{N}, {K}] -> packed [{result['weight_packed'].shape[0]}, {result['weight_packed'].shape[1]}]")

    # ---- Correctness ----
    print("\n" + "=" * 78)
    print("CORRECTNESS CHECK  (per-layer cosine thresholds)")
    print("=" * 78)

    all_passed = True
    for name in layer_names:
        if name not in activations:
            print(f"  {name:15s}  SKIPPED (no activation data)")
            continue

        act = activations[name]
        wgt = weights[name]
        qw = quantized_weights[name]
        M, K = act.shape
        N = wgt.shape[0]
        threshold = COSINE_THRESHOLDS.get(name, COSINE_THRESHOLD_DEFAULT)

        passed, cos, ref_cos = check_correctness(
            ref, sol, act, wgt,
            qw["weight_packed"], qw["weight_scales"], gs,
            threshold=threshold,
        )
        status = "PASS" if passed else "FAIL"
        print(f"  {name:15s} ({M}x{N}x{K}):  cosine={cos:.6f}  threshold={threshold:.3f}  [{status}]", end="")
        if ref_cos < 0.99:
            print(f"  (ref cuda: {ref_cos:.4f} WARNING)")
        else:
            print()

        if not passed:
            all_passed = False

    if not all_passed:
        print("\nFAILED: Your solution did not pass correctness checks.")
        print("Fix your kernel/quantize.py and try again.")
        sys.exit(1)

    print("\nAll correctness checks passed.")

    # ---- Performance Benchmark ----
    print("\n" + "=" * 78)
    print("PERFORMANCE BENCHMARK")
    print("=" * 78)

    header = f"  {'Layer':15s} {'M':>5s} {'N':>6s} {'K':>6s}  {'Quant GB/s':>11s}  {'GEMM TOPs':>10s}  {'Ref TOPs':>10s}  {'Speedup':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    score_gemm_tops = []

    for name in layer_names:
        if name not in activations:
            continue

        act = activations[name]
        wgt = weights[name]
        qw = quantized_weights[name]
        M, K = act.shape
        N = wgt.shape[0]

        quant_gbs, gemm_tops, ref_gemm_tops = run_benchmark(
            ref, sol, act, wgt,
            qw["weight_packed"], qw["weight_scales"], gs,
            warmup=args.warmup, iters=args.iters,
        )
        speedup = gemm_tops / ref_gemm_tops if ref_gemm_tops > 0 else 0

        print(f"  {name:15s} {M:5d} {N:6d} {K:6d}  {quant_gbs:>9.2f}    {gemm_tops:>8.2f}    {ref_gemm_tops:>8.2f}    {speedup:>6.2f}x")
        score_gemm_tops.append(gemm_tops)

    # ---- Score ----
    print("\n" + "=" * 78)
    if score_gemm_tops:
        avg_score = sum(score_gemm_tops) / len(score_gemm_tops)
        print(f"SCORE: Avg GEMM TOPs = {avg_score:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
