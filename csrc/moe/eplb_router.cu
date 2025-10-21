#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

using i64 = int64_t;

#define CUDA_CHECK(expr) do { auto _e = (expr); if (_e != cudaSuccess) { printf("CUDA error %s @ %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); asm("trap;"); } } while(0)

// ---------------- Greedy (all int64) ----------------
__global__ void greedy_kernel(
    const i64* __restrict__ off,     // [n+1]
    const i64* __restrict__ idx,     // [nnz]
    i64* __restrict__ chosen,        // [n]
    i64* __restrict__ rank_loads,    // [P]
    i64 n)
{
  for (i64 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (i64)blockDim.x * gridDim.x) {
    i64 s = off[e], t = off[e+1];
    i64 best_p = -1, best_l = LLONG_MAX;
    for (i64 j = s; j < t; ++j) {
      i64 p = idx[j];
      i64 l = __ldg(rank_loads + p);
      if (l < best_l || (l == best_l && p < best_p)) { best_l = l; best_p = p; }
    }
    #pragma unroll 4
    for (int attempt = 0; attempt < 4; ++attempt) {
      i64 expected = __ldg(rank_loads + best_p);
      // 64-bit CAS uses unsigned long long
      auto addr = reinterpret_cast<unsigned long long*>(rank_loads + best_p);
      unsigned long long old = atomicCAS(addr, (unsigned long long)expected, (unsigned long long)(expected + 1));
      if ((i64)old == expected) { chosen[e] = best_p; break; }
      best_p = -1; best_l = LLONG_MAX;
      for (i64 j = s; j < t; ++j) {
        i64 p = idx[j];
        i64 l = __ldg(rank_loads + p);
        if (l < best_l || (l == best_l && p < best_p)) { best_l = l; best_p = p; }
      }
      if (attempt == 3) { atomicAdd(reinterpret_cast<unsigned long long*>(rank_loads + best_p),
          static_cast<unsigned long long>(1)); chosen[e] = best_p; }
    }
  }
}

static inline int launch_blocks_1d(i64 n, int threads=256, int maxb=1024){
  return (int)std::min((n + threads - 1)/threads, (i64)maxb);
}

void greedy_smallest_choice_first_cuda(
    const at::Tensor& off, const at::Tensor& idx, at::Tensor& chosen, i64 P)
{
  TORCH_CHECK(off.is_cuda() && idx.is_cuda() && chosen.is_cuda(), "CUDA tensors required");
  const i64 n = (i64)off.size(0) - 1;
  auto loads = at::zeros({P}, off.options().dtype(at::kLong));
  const int threads = 256;
  const int blocks  = launch_blocks_1d(n, threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  greedy_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const i64*>(off.data_ptr()), static_cast<const i64*>(idx.data_ptr()), static_cast<i64*>(chosen.data_ptr()), static_cast<i64*>(loads.data_ptr()), n);
}

// --------------- Exact (all int64) ---------------
__global__ void build_slot_degrees_kernel(const i64* __restrict__ off, i64* __restrict__ degL, i64 n, i64 L) {
  for (i64 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (i64)blockDim.x * gridDim.x) {
    degL[e] = (off[e+1] - off[e]) * L;
  }
}

__global__ void expand_e2slot_kernel(const i64* __restrict__ off, const i64* __restrict__ idx,
                                     const i64* __restrict__ slot_off, i64* __restrict__ slot_idx,
                                     i64 n, i64 L) {
  for (i64 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (i64)blockDim.x * gridDim.x) {
    i64 out = slot_off[e];
    for (i64 j = off[e]; j < off[e+1]; ++j) {
      i64 r = idx[j];
      for (i64 s = 0; s < L; ++s) slot_idx[out++] = r * L + s;
    }
  }
}

__global__ void try_match_kernel(const i64* __restrict__ off, const i64* __restrict__ idx,
                                 i64* __restrict__ pairU, i64* __restrict__ pairV, i64 n) {
  for (i64 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (i64)blockDim.x * gridDim.x) {
    if (pairU[e] != -1) continue;
    for (i64 j = off[e]; j < off[e+1]; ++j) {
      i64 v = idx[j];
      auto addr = reinterpret_cast<unsigned long long*>(pairV + v);
      long long prev = (long long)atomicCAS(addr, (unsigned long long)(-1LL), (unsigned long long)e);
      if (prev == -1) { pairU[e] = v; break; }
    }
  }
}

__global__ void clear_if_conflict_kernel(i64* __restrict__ pairU, const i64* __restrict__ pairV, i64 n) {
  for (i64 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (i64)blockDim.x * gridDim.x) {
    i64 v = pairU[e];
    if (v != -1 && pairV[v] != e) pairU[e] = -1;
  }
}

__global__ void fill_ranks_from_slots_kernel(const i64* __restrict__ pairU, i64* __restrict__ ranks, i64 n, i64 L) {
  for (i64 e = blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (i64)blockDim.x * gridDim.x) {
    i64 slot = pairU[e];
    ranks[e] = (slot >= 0) ? (slot / L) : -1;
  }
}

static void device_build_e2slot_csr(const at::Tensor& off, const at::Tensor& idx, i64 L,
                                    at::Tensor& slot_off, at::Tensor& slot_idx)
{
  const i64 n = (i64)off.size(0) - 1;
  auto opts_i64 = off.options().dtype(at::kLong);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocks  = launch_blocks_1d(n, threads);

  auto degL = at::empty({n}, opts_i64);
  build_slot_degrees_kernel<<<blocks, threads, 0, stream>>>(static_cast<const i64*>(off.data_ptr()), static_cast<i64*>(degL.data_ptr()), n, L);

  // slot_off = cat([0], cumsum(degL))  // all long
  slot_off = at::cat({at::zeros({1}, opts_i64), degL.cumsum(0)}, 0);
  const i64 nnzL = slot_off.select(0, n).item<i64>();

  slot_idx = at::empty({nnzL}, opts_i64);
  expand_e2slot_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const i64*>(off.data_ptr()), static_cast<const i64*>(idx.data_ptr()), static_cast<const i64*>(slot_off.data_ptr()), static_cast<i64*>(slot_idx.data_ptr()), n, L);
}

at::Tensor exact_min_max_activations_cuda(
    const at::Tensor& off, const at::Tensor& idx, at::Tensor& chosen_rank, i64 P)
{
  TORCH_CHECK(off.is_cuda() && idx.is_cuda() && chosen_rank.is_cuda(), "CUDA tensors required");
  const i64 n = (i64)off.size(0) - 1;
  i64 low = (n + P - 1) / P, high = n, best = n;

  auto opts_i64 = off.options().dtype(at::kLong);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int blocksU = launch_blocks_1d(n, threads);

  const int MAX_BS_ITERS = 32;
  const int HK_ITERS     = 8;

  at::Tensor e_off, e_idx, pairU, pairV, ranks;

  for (int it = 0; it < MAX_BS_ITERS; ++it) {
    i64 mid = (low + high) >> 1;
    if (mid < low || mid > high) mid = low;

    device_build_e2slot_csr(off, idx, mid, e_off, e_idx);
    const i64 Vslots = (i64)(P * mid);
    pairU = at::full({n}, (i64)-1, opts_i64);
    pairV = at::full({Vslots}, (i64)-1, opts_i64);

    for (int k = 0; k < HK_ITERS; ++k) {
      try_match_kernel<<<blocksU, threads, 0, stream>>>(static_cast<const i64*>(e_off.data_ptr()), static_cast<const i64*>(e_idx.data_ptr()),
                                                        static_cast<i64*>(pairU.data_ptr()), static_cast<i64*>(pairV.data_ptr()), n);
      clear_if_conflict_kernel<<<blocksU, threads, 0, stream>>>( static_cast<i64*>(pairU.data_ptr()), static_cast<i64*>(pairV.data_ptr()), n);
    }

    auto matched_t = (pairU != -1).sum();        // long scalar
    i64 matched = matched_t.item<i64>();

    if (matched == n) {
      best = mid; high = mid - 1;
      ranks = at::empty_like(chosen_rank);
      fill_ranks_from_slots_kernel<<<blocksU, threads, 0, stream>>>(static_cast<const i64*>(pairU.data_ptr()) , static_cast<i64*>(ranks.data_ptr()) , n, mid);
      chosen_rank.copy_(ranks);
    } else {
      low = mid + 1;
    }
    if (low > high) break;
  }
  return at::scalar_tensor(best, opts_i64);
}

// --------------- Replica selection + token mapping (all int64) ---------------
__global__ void pick_replica_kernel_ok(
    const i64* __restrict__ l2p, const i64* __restrict__ lrc,
    const i64* __restrict__ active, const i64* __restrict__ chosen_rank,
    i64* __restrict__ chosen_replica, i64 n, i64 Rmax, i64 P)
{
  for (i64 i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (i64)blockDim.x * gridDim.x) {
    i64 e = active[i];
    i64 want_p = chosen_rank[i];
    i64 rnum = lrc[e];
    i64 pick = -1;
    for (i64 r = 0; r < rnum; ++r) {
      i64 phys = l2p[e * Rmax + r];
      i64 p = phys % P; // replace if explicit phys2rank exists
      if (p == want_p) { pick = phys; break; }
    }
    chosen_replica[i] = pick;
  }
}

void select_replica_on_rank_cuda(
    const at::Tensor& l2p, const at::Tensor& lrc, const at::Tensor& active,
    const at::Tensor& chosen_rank, at::Tensor& chosen_replica, i64 P)
{
  const i64 n = (i64)active.size(0);
  const int threads = 256;
  const int blocks  = launch_blocks_1d(n, threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  pick_replica_kernel_ok<<<blocks, threads, 0, stream>>>(
      static_cast<const i64*>(l2p.data_ptr()), static_cast<const i64*>(lrc.data_ptr()), static_cast<const i64*>(active.data_ptr()),
      static_cast<const i64*>(chosen_rank.data_ptr()), static_cast<i64*>(chosen_replica.data_ptr()),
      n, (i64)l2p.size(1), P);
}

void map_tokens_to_chosen_replica_cuda(
    const at::Tensor& topk_ids_logical, const at::Tensor& active,
    const at::Tensor& chosen_replica, at::Tensor& out_physical_ids)
{
  const i64 E = 1 + at::max(active).item<i64>();
  auto opts = topk_ids_logical.options().dtype(at::kLong);
  auto lut = at::full({E}, (i64)-1, opts);
  lut.index_put_({active}, chosen_replica);

  auto flat_in  = topk_ids_logical.reshape({-1}).contiguous();
  auto gathered = lut.index_select(0, flat_in);
  out_physical_ids.copy_(gathered.reshape_as(topk_ids_logical));
}
