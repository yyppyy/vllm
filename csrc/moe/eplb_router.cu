#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/full.h>
#include <ATen/ops/max.h>
#include <ATen/ops/zeros.h>
#include <cuda_runtime.h>
#include <limits>

using i32 = int32_t;
using i64 = int64_t;

#define CUDA_CHECK(expr) do { auto _e = (expr); if (_e != cudaSuccess) { printf("CUDA error %s @ %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); asm("trap;"); } } while(0)

// ---------------- Greedy: smallest-choice-first ----------------
__global__ void greedy_kernel(
    const i32* __restrict__ off,   // [n+1]
    const i32* __restrict__ idx,   // [nnz]
    i32* __restrict__ chosen,      // [n]
    i32* __restrict__ rank_loads,  // [P]
    i32 n)
{
  for (i32 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
    i32 s = off[e], t = off[e+1];
    i32 best_p = -1, best_l = INT_MAX;
    for (i32 j = s; j < t; ++j) {
      i32 p = idx[j];
      i32 l = __ldg(rank_loads + p);
      if (l < best_l || (l == best_l && p < best_p)) { best_l = l; best_p = p; }
    }
    #pragma unroll 4
    for (int attempt = 0; attempt < 4; ++attempt) {
      i32 expected = __ldg(rank_loads + best_p);
      i32 old = atomicCAS(rank_loads + best_p, expected, expected + 1);
      if (old == expected) { chosen[e] = best_p; break; }
      best_p = -1; best_l = INT_MAX;
      for (i32 j = s; j < t; ++j) {
        i32 p = idx[j];
        i32 l = __ldg(rank_loads + p);
        if (l < best_l || (l == best_l && p < best_p)) { best_l = l; best_p = p; }
      }
      if (attempt == 3) { atomicAdd(rank_loads + best_p, 1); chosen[e] = best_p; }
    }
  }
}

void greedy_smallest_choice_first_cuda(
    const at::Tensor& off, const at::Tensor& idx, at::Tensor& chosen, int32_t P)
{
  TORCH_CHECK(off.is_cuda() && idx.is_cuda() && chosen.is_cuda(), "CUDA tensors required");
  const i32 n = (i32)off.size(0) - 1;
  auto loads = at::zeros({P}, off.options().dtype(at::kInt));
  const int threads = 256;
  const int blocks  = std::min((n + threads - 1)/threads, 1024);
  auto stream = at::cuda::getCurrentCUDAStream();
  greedy_kernel<<<blocks, threads, 0, stream>>>(off.data_ptr<i32>(), idx.data_ptr<i32>(),
                                                chosen.data_ptr<i32>(), loads.data_ptr<i32>(), n);
}

// --------------- Exact: device-only CSR + fixed-iter matching ---------------
__global__ void build_slot_degrees_kernel(const i32* __restrict__ off, i32* __restrict__ degL, int n, int L) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
    degL[e] = (off[e+1] - off[e]) * L;
  }
}

__global__ void expand_e2slot_kernel(const i32* __restrict__ off, const i32* __restrict__ idx,
                                     const i32* __restrict__ slot_off, i32* __restrict__ slot_idx,
                                     int n, int L) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
    int out = slot_off[e];
    for (int j = off[e]; j < off[e+1]; ++j) {
      int r = idx[j];
      for (int s = 0; s < L; ++s) slot_idx[out++] = r * L + s;
    }
  }
}

__global__ void try_match_kernel(const i32* __restrict__ off, const i32* __restrict__ idx,
                                 i32* __restrict__ pairU, i32* __restrict__ pairV, int n) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
    if (pairU[e] != -1) continue;
    for (int j = off[e]; j < off[e+1]; ++j) {
      int v = idx[j];
      int prev = atomicCAS(&pairV[v], -1, e);
      if (prev == -1) { pairU[e] = v; break; }
    }
  }
}

__global__ void clear_if_conflict_kernel(i32* __restrict__ pairU, const i32* __restrict__ pairV, int n) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
    int v = pairU[e];
    if (v != -1 && pairV[v] != e) pairU[e] = -1;
  }
}

__global__ void fill_ranks_from_slots_kernel(const i32* __restrict__ pairU, i32* __restrict__ ranks, int n, int L) {
  for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += blockDim.x * gridDim.x) {
    int slot = pairU[e];
    ranks[e] = (slot >= 0) ? (slot / L) : -1;
  }
}

static void device_build_e2slot_csr(const at::Tensor& off, const at::Tensor& idx, int L,
                                    at::Tensor& slot_off, at::Tensor& slot_idx)
{
  const int n = (int)off.size(0) - 1;
  auto opts_i32 = off.options().dtype(at::kInt);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocks  = std::min((n + threads - 1)/threads, 1024);

  auto degL = at::empty({n}, opts_i32);
  build_slot_degrees_kernel<<<blocks, threads, 0, stream>>>(off.data_ptr<i32>(), degL.data_ptr<i32>(), n, L);

  // slot_off = cat([0], cumsum(degL))
  slot_off = at::cat({at::zeros({1}, opts_i32), degL.cumsum(0)}, 0);
  const int nnzL = slot_off.select(0, n).item<i32>();  // one scalar read inside C++ op is fine

  slot_idx = at::empty({nnzL}, opts_i32);
  expand_e2slot_kernel<<<blocks, threads, 0, stream>>>(off.data_ptr<i32>(), idx.data_ptr<i32>(),
                                                       slot_off.data_ptr<i32>(), slot_idx.data_ptr<i32>(), n, L);
}

at::Tensor exact_min_max_activations_cuda(
    const at::Tensor& off, const at::Tensor& idx, at::Tensor& chosen_rank, int32_t P)
{
  TORCH_CHECK(off.is_cuda() && idx.is_cuda() && chosen_rank.is_cuda(), "CUDA tensors required");
  const int n = (int)off.size(0) - 1;
  int low = (n + P - 1) / P, high = n, best = n;

  auto opts_i32 = off.options().dtype(at::kInt);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocksU = std::min((n + threads - 1)/threads, 1024);

  const int MAX_BS_ITERS = 32;     // fixed outer iters (stable launches)
  const int HK_ITERS     = 8;      // fixed inner rounds

  at::Tensor e_off, e_idx, pairU, pairV, ranks;

  for (int it = 0; it < MAX_BS_ITERS; ++it) {
    int mid = (low + high) >> 1;
    if (mid < low || mid > high) mid = low;

    // Build expanded graph for this L
    device_build_e2slot_csr(off, idx, mid, e_off, e_idx);
    const int Vslots = (int)(P * mid);
    pairU = at::full({n}, -1, opts_i32);
    pairV = at::full({Vslots}, -1, opts_i32);

    for (int k = 0; k < HK_ITERS; ++k) {
      try_match_kernel<<<blocksU, threads, 0, stream>>>(e_off.data_ptr<i32>(), e_idx.data_ptr<i32>(),
                                                        pairU.data_ptr<i32>(), pairV.data_ptr<i32>(), n);
      clear_if_conflict_kernel<<<blocksU, threads, 0, stream>>>(pairU.data_ptr<i32>(), pairV.data_ptr<i32>(), n);
    }

    auto matched_t = (pairU != -1).sum();   // Tensor scalar
    int matched = matched_t.item<int>();    // host read inside C++ op (launch pattern is fixed)

    if (matched == n) {
      best = mid; high = mid - 1;
      ranks = at::empty_like(chosen_rank);
      fill_ranks_from_slots_kernel<<<blocksU, threads, 0, stream>>>(pairU.data_ptr<i32>(), ranks.data_ptr<i32>(), n, mid);
      chosen_rank.copy_(ranks);
    } else {
      low = mid + 1;
    }
    if (low > high) break;
  }
  return at::scalar_tensor(best, opts_i32);
}

// --------------- Replica selection + token mapping (device-only) ---------------
__global__ void pick_replica_kernel_ok(
    const i64* __restrict__ l2p, const i32* __restrict__ lrc,
    const i64* __restrict__ active, const i32* __restrict__ chosen_rank,
    i64* __restrict__ chosen_replica, int n, int Rmax, int P)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    i64 e = active[i];
    int want_p = chosen_rank[i];
    int rnum = lrc[e];
    i64 pick = -1;
    for (int r = 0; r < rnum; ++r) {
      i64 phys = l2p[e * (i64)Rmax + r];
      int p = (int)(phys % P); // replace if explicit phys2rank exists
      if (p == want_p) { pick = phys; break; }
    }
    chosen_replica[i] = pick;
  }
}

void select_replica_on_rank_cuda(
    const at::Tensor& l2p, const at::Tensor& lrc, const at::Tensor& active,
    const at::Tensor& chosen_rank, at::Tensor& chosen_replica, int32_t P)
{
  const int n = (int)active.size(0);
  const int threads = 256;
  const int blocks  = std::min((n + threads - 1)/threads, 1024);
  auto stream = at::cuda::getCurrentCUDAStream();
  pick_replica_kernel_ok<<<blocks, threads, 0, stream>>>(
      l2p.data_ptr<i64>(), lrc.data_ptr<i32>(), active.data_ptr<i64>(),
      chosen_rank.data_ptr<i32>(), chosen_replica.data_ptr<i64>(),
      n, (int)l2p.size(1), P);
}

void map_tokens_to_chosen_replica_cuda(
    const at::Tensor& topk_ids_logical, const at::Tensor& active,
    const at::Tensor& chosen_replica, at::Tensor& out_physical_ids)
{
  // Build dense LUT on device: lut[logical] = replica
  const i64 E = 1 + at::max(active).item<i64>();    // scalar read inside C++ op
  auto opts = topk_ids_logical.options().dtype(at::kLong);
  auto lut = at::full({E}, (i64)-1, opts);
  // scatter: lut[active] = chosen_replica
  lut.index_put_({active}, chosen_replica);

  // gather: out = lut[topk_ids_logical]
  auto flat_in  = topk_ids_logical.reshape({-1}).contiguous();
  auto gathered = lut.index_select(0, flat_in);
  auto flat_out = gathered.reshape_as(flat_in);
  out_physical_ids.copy_(flat_out.reshape_as(topk_ids_logical));
}
