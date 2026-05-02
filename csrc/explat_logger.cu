// Per-(rank, layer, batch) MoE profiler kernels.
//
// Two ops:
//   record_stamp(stamps, idx)
//     1-thread kernel that reads %globaltimer and writes the current
//     GPU nanosecond timestamp into stamps[idx].
//
//   log_expert_tokens(rank, layer_idx, M, expert_num_tokens, stamps,
//                     armed, counter, ringbuf, e_max)
//     1-warp kernel. If armed[0]==0, returns early. Otherwise atomically
//     increments counter[0], picks slot = seq % n_slots, and writes one
//     record to a UVA-mapped pinned-host ringbuffer:
//       [seq, rank, layer, M, num_local_experts,
//        align_ns, gemm_gu_ns, silu_ns, quant_ns, gemm_dn_ns,
//        expert_tokens[e_max]]
//     Issues __threadfence_system() so host reads see the writes.
//
// Both kernels are captureable into CUDA graphs.

#include <torch/all.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace vllm {
namespace explat {

// Slot layout (ints, packed into int64 ringbuffer for natural alignment):
//   int64 seq
//   int32 rank, int32 layer_idx          (one int64)
//   int32 M, int32 num_local_experts     (one int64)
//   int64 align_ns
//   int64 gemm_gu_ns
//   int64 silu_ns
//   int64 quant_ns
//   int64 gemm_dn_ns
// followed by e_max int32 expert_tokens (packed two-per-int64).
// Caller passes slot_stride_int64 = 7 + (e_max + 1) / 2.
//
// We compute slot_stride_int64 host-side from ringbuf.size(1).

__global__ void record_stamp_kernel(int64_t* __restrict__ stamps,
                                    int idx) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    stamps[idx] = (int64_t)t;
  }
}

__global__ void log_expert_tokens_kernel(
    int rank,
    int layer_idx,
    int M,
    const int32_t* __restrict__ expert_num_tokens,
    int num_local_experts,
    const int64_t* __restrict__ stamps,
    const int32_t* __restrict__ armed,
    int32_t* __restrict__ counter,
    int64_t* __restrict__ ringbuf,
    int n_slots,
    int slot_stride_int64,
    int e_max) {
  int lane = threadIdx.x;

  __shared__ int s_seq;
  __shared__ int s_skip;

  if (lane == 0) {
    if (armed[0] == 0) {
      s_skip = 1;
    } else {
      s_skip = 0;
      s_seq = atomicAdd(counter, 1);
    }
  }
  __syncthreads();
  if (s_skip) return;

  int slot_idx = s_seq % n_slots;
  int64_t* slot = ringbuf + (int64_t)slot_idx * slot_stride_int64;

  if (lane == 0) {
    // Compute deltas (clamped to 0 in case of any reordering).
    int64_t s0 = stamps[0];
    int64_t s1 = stamps[1];
    int64_t s2 = stamps[2];
    int64_t s3 = stamps[3];
    int64_t s4 = stamps[4];
    int64_t s5 = stamps[5];
    int64_t align_ns = s1 > s0 ? s1 - s0 : 0;
    int64_t gemm_gu_ns = s2 > s1 ? s2 - s1 : 0;
    int64_t silu_ns = s3 > s2 ? s3 - s2 : 0;
    int64_t quant_ns = s4 > s3 ? s4 - s3 : 0;
    int64_t gemm_dn_ns = s5 > s4 ? s5 - s4 : 0;

    slot[0] = (int64_t)s_seq;
    // Pack rank, layer_idx into one int64 (low: rank, high: layer)
    slot[1] = ((int64_t)(uint32_t)layer_idx << 32) |
              ((int64_t)(uint32_t)rank);
    // Pack M, num_local_experts
    slot[2] = ((int64_t)(uint32_t)num_local_experts << 32) |
              ((int64_t)(uint32_t)M);
    slot[3] = align_ns;
    slot[4] = gemm_gu_ns;
    slot[5] = silu_ns;
    slot[6] = quant_ns;
    slot[7] = gemm_dn_ns;
  }
  __syncthreads();

  // Lane-parallel copy of expert_num_tokens, packed two int32s per int64.
  // Slot[8 + i] holds tokens[2*i] in low 32 bits and tokens[2*i+1] in
  // high 32 bits.
  int n_pairs = (e_max + 1) / 2;
  int n = num_local_experts;
  for (int i = lane; i < n_pairs; i += blockDim.x) {
    int j0 = 2 * i;
    int j1 = 2 * i + 1;
    uint32_t v0 = (j0 < n) ? (uint32_t)expert_num_tokens[j0] : 0u;
    uint32_t v1 = (j1 < n) ? (uint32_t)expert_num_tokens[j1] : 0u;
    int64_t packed = ((int64_t)v1 << 32) | (int64_t)v0;
    slot[8 + i] = packed;
  }
  __syncthreads();
  // Make all writes (including those to UVA-mapped pinned host memory)
  // visible to the host.
  if (lane == 0) {
    __threadfence_system();
  }
}

void record_stamp(torch::Tensor stamps, int64_t idx) {
  TORCH_CHECK(stamps.dtype() == at::kLong);
  TORCH_CHECK(stamps.is_cuda());
  TORCH_CHECK(stamps.is_contiguous());
  TORCH_CHECK(idx >= 0 && idx < stamps.numel());
  const at::cuda::OptionalCUDAGuard guard(stamps.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  record_stamp_kernel<<<1, 1, 0, stream>>>(
      stamps.data_ptr<int64_t>(), static_cast<int>(idx));
}

void log_expert_tokens(int64_t rank,
                       int64_t layer_idx,
                       int64_t M,
                       torch::Tensor expert_num_tokens,
                       torch::Tensor stamps,
                       torch::Tensor armed,
                       torch::Tensor counter,
                       torch::Tensor ringbuf,
                       int64_t e_max) {
  TORCH_CHECK(expert_num_tokens.dtype() == at::kInt);
  TORCH_CHECK(stamps.dtype() == at::kLong);
  TORCH_CHECK(armed.dtype() == at::kInt);
  TORCH_CHECK(counter.dtype() == at::kInt);
  TORCH_CHECK(ringbuf.dtype() == at::kLong);
  TORCH_CHECK(expert_num_tokens.is_cuda());
  TORCH_CHECK(stamps.is_cuda());
  TORCH_CHECK(armed.is_cuda());
  TORCH_CHECK(counter.is_cuda());
  // ringbuf is a pinned host tensor; UVA makes its data_ptr usable from
  // the GPU. We do not require .is_cuda(), only contiguity.
  TORCH_CHECK(ringbuf.is_contiguous());
  TORCH_CHECK(ringbuf.dim() == 2);
  TORCH_CHECK(stamps.numel() == 6);

  int n_slots = static_cast<int>(ringbuf.size(0));
  int slot_stride_int64 = static_cast<int>(ringbuf.size(1));
  int em = static_cast<int>(e_max);
  TORCH_CHECK(slot_stride_int64 >= 8 + (em + 1) / 2);

  const at::cuda::OptionalCUDAGuard guard(stamps.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  log_expert_tokens_kernel<<<1, 32, 0, stream>>>(
      static_cast<int>(rank),
      static_cast<int>(layer_idx),
      static_cast<int>(M),
      expert_num_tokens.data_ptr<int32_t>(),
      static_cast<int>(expert_num_tokens.numel()),
      stamps.data_ptr<int64_t>(),
      armed.data_ptr<int32_t>(),
      counter.data_ptr<int32_t>(),
      static_cast<int64_t*>(ringbuf.data_ptr()),
      n_slots,
      slot_stride_int64,
      em);
}

// ---------------------------------------------------------------------
// Breakdown logger
// ---------------------------------------------------------------------
//
// log_breakdown(rank, layer_idx, M,
//               py_stamps,        // int64[5]: attn_start, attn_end,
//                                 //          gate_start, gate_end,
//                                 //          dispatch_end
//               expert_stamps,    // int64[6]: align_s/e, gemm_gu, silu,
//                                 //          quant, gemm_dn
//               dc_stamps,        // int64[37]: dispatch_combine
//                                 //          profiling_timestamps
//                                 //          (kDarNumSteps=19 +
//                                 //           kCasNumSteps=18)
//               armed, counter, ringbuf)
//
// Reads the in-kernel %globaltimer values that DC_TIMESTAMP and
// record_stamp wrote during the layer's MoE forward, computes 6
// per-(rank, layer, batch) deltas, and writes one ringbuffer slot.
//
// Slot layout (int64s):
//   [0] seq
//   [1] (rank low32 | layer_idx high32)
//   [2] M (low32; high32 unused)
//   [3] attention_ns
//   [4] gating_ns
//   [5] routing_ns
//   [6] dispatch_ns
//   [7] expert_ns
//   [8] combine_ns
// slot_stride_int64 = 9.

namespace breakdown_constants {
constexpr int kDarReadCounters = 0;
constexpr int kDarScanClaim    = 4;
constexpr int kDarEnd          = 18;
constexpr int kCasStart        = 19 + 0;   // cas:read_counters
// `cas:end` lives at slot 16; slot 17 (`cas:?`) is reserved and
// never written by combine_and_scatter_kernel — reading it yields
// zero, which used to make combine_ns always 0.
constexpr int kCasEnd          = 19 + 16;  // cas:end
}

__global__ void log_breakdown_kernel(
    int rank,
    int layer_idx,
    int M,
    const int64_t* __restrict__ py_stamps,
    const int64_t* __restrict__ expert_stamps,
    const int64_t* __restrict__ dc_stamps,
    const int32_t* __restrict__ armed,
    int32_t* __restrict__ counter,
    int64_t* __restrict__ ringbuf,
    int n_slots,
    int slot_stride_int64) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  if (armed[0] == 0) return;

  using namespace breakdown_constants;

  int seq = atomicAdd(counter, 1);
  int slot_idx = seq % n_slots;
  int64_t* slot = ringbuf + (int64_t)slot_idx * slot_stride_int64;

  auto pos_diff = [](int64_t a, int64_t b) -> int64_t {
    return a > b ? a - b : 0;
  };

  // Python-side category boundaries.
  int64_t attn_start  = py_stamps[0];
  int64_t attn_end    = py_stamps[1];
  int64_t gate_start  = py_stamps[2];
  int64_t gate_end    = py_stamps[3];
  int64_t dispatch_end = py_stamps[4];

  // In-kernel timestamps.
  int64_t dar_start  = dc_stamps[kDarReadCounters];
  int64_t dar_route  = dc_stamps[kDarScanClaim];
  int64_t dar_end    = dc_stamps[kDarEnd];
  (void)dar_end;  // dispatch is computed via py_stamps minus routing.
  int64_t cas_start  = dc_stamps[kCasStart];
  int64_t cas_end    = dc_stamps[kCasEnd];

  int64_t attention_ns  = pos_diff(attn_end, attn_start);
  int64_t gating_ns     = pos_diff(gate_end, gate_start);
  int64_t routing_ns    = pos_diff(dar_route, dar_start);
  int64_t dispatch_full = pos_diff(dispatch_end, gate_end);
  int64_t dispatch_ns   = dispatch_full > routing_ns
                          ? dispatch_full - routing_ns : 0;
  int64_t expert_ns     = pos_diff(expert_stamps[5], expert_stamps[0]);
  int64_t combine_ns    = pos_diff(cas_end, cas_start);

  slot[0] = (int64_t)seq;
  slot[1] = ((int64_t)(uint32_t)layer_idx << 32) |
            ((int64_t)(uint32_t)rank);
  slot[2] = (int64_t)(uint32_t)M;
  slot[3] = attention_ns;
  slot[4] = gating_ns;
  slot[5] = routing_ns;
  slot[6] = dispatch_ns;
  slot[7] = expert_ns;
  slot[8] = combine_ns;

  __threadfence_system();
}

void log_breakdown(int64_t rank,
                   int64_t layer_idx,
                   int64_t M,
                   torch::Tensor py_stamps,
                   torch::Tensor expert_stamps,
                   torch::Tensor dc_stamps,
                   torch::Tensor armed,
                   torch::Tensor counter,
                   torch::Tensor ringbuf) {
  TORCH_CHECK(py_stamps.dtype()     == at::kLong);
  TORCH_CHECK(expert_stamps.dtype() == at::kLong);
  TORCH_CHECK(dc_stamps.dtype()     == at::kLong);
  TORCH_CHECK(armed.dtype()         == at::kInt);
  TORCH_CHECK(counter.dtype()       == at::kInt);
  TORCH_CHECK(ringbuf.dtype()       == at::kLong);
  TORCH_CHECK(py_stamps.is_cuda());
  TORCH_CHECK(expert_stamps.is_cuda());
  TORCH_CHECK(dc_stamps.is_cuda());
  TORCH_CHECK(armed.is_cuda());
  TORCH_CHECK(counter.is_cuda());
  TORCH_CHECK(ringbuf.is_contiguous());
  TORCH_CHECK(ringbuf.dim() == 2);
  TORCH_CHECK(py_stamps.numel()     == 5);
  TORCH_CHECK(expert_stamps.numel() == 6);
  TORCH_CHECK(dc_stamps.numel()     == 37);

  int n_slots = static_cast<int>(ringbuf.size(0));
  int slot_stride_int64 = static_cast<int>(ringbuf.size(1));
  TORCH_CHECK(slot_stride_int64 >= 9);

  const at::cuda::OptionalCUDAGuard guard(py_stamps.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  log_breakdown_kernel<<<1, 1, 0, stream>>>(
      static_cast<int>(rank),
      static_cast<int>(layer_idx),
      static_cast<int>(M),
      py_stamps.data_ptr<int64_t>(),
      expert_stamps.data_ptr<int64_t>(),
      dc_stamps.data_ptr<int64_t>(),
      armed.data_ptr<int32_t>(),
      counter.data_ptr<int32_t>(),
      static_cast<int64_t*>(ringbuf.data_ptr()),
      n_slots,
      slot_stride_int64);
}

}  // namespace explat
}  // namespace vllm
