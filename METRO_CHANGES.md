# METRO fork — change notes

Branch `v0.11.0-gcp-metro-haiyue`, forked from `v0.11.0-gcp-metro`.

These commits do three separable things: make the tree build and run on
Princeton's Della cluster, make the dispatch kernel observable at the phase
level, and remove one measurable inefficiency that the new observability
exposed. A fourth group fixes two benchmark artifacts that were larger than
the effects under study.

Every performance claim below was measured on 8×A100-SXM4-40GB (NVSwitch,
NV12 between every pair) with Qwen3-30B-A3B (128 experts, top-8) and
ERNIE-4.5-21B-A3B-PT (64 experts, top-6).

---

## 1. `build:` make the fork buildable and runnable on Della

Four independent blockers, none of them code bugs — all are environment
assumptions that no longer hold.

| Blocker | Symptom | Fix |
| --- | --- | --- |
| GCC 11.5 cannot compile PyTorch 2.8 headers | `ATen/core/List_inl.h: need 'typename' before ... dependent scope` | pin `gcc-toolset-13` when present |
| CMake picks `/usr/local/cuda` (13.3) | mismatches the cu128 torch wheel | pin `/usr/local/cuda-12.8` |
| `--torch-backend=auto` on a GPU-less login node | installs the wrong wheel | use `cu128` explicitly; set `TORCH_CUDA_ARCH_LIST=8.0` so configure needs no visible GPU |
| `huggingface-cli` is deprecated upstream | exits without downloading | switch both download scripts to `hf` |
| Della's resolver returns unreachable CloudFront edges for `huggingface.co` | every download hangs on TCP connect | `tools/net/dns8888.py` re-resolves via 8.8.8.8 |

Each pin is guarded by `if [[ -d ... ]]`, so on a host without these paths the
behaviour is unchanged. The DNS override applies only to `*.huggingface.co`
and `*.hf.co`; everything else uses the system resolver.

Not fixed here but worth recording: Della's compute nodes have no route to the
internet, so wheels, models and datasets must all be fetched on a login node
before submitting a job.

**Effect on results: none.** This only affects whether the tree runs.

---

## 2. `profile:` record all 19 dispatch-kernel timestamps, not 4

`dispatch_and_route_kernel` already wrote 19 `DC_TIMESTAMP` marks covering
every phase, but `log_breakdown` read only four of them and derived two
numbers:

```c
routing_ns  = dc_stamps[4] - dc_stamps[0];
dispatch_ns = (py_stamps[4] - py_stamps[3]) - routing_ns;
```

Two consequences follow, and both change how the existing figures should be
read.

**`routing_ns` does not measure METRO's routing algorithm.** Slots 0–4 are
`read_counters → scan_write → scan_expand → scan_group → scan_claim`: per-token
top-k expansion and grouping by destination rank, which *both* routing modes
execute identically. Algorithm 1 runs in slots 13–16 and therefore lands inside
`dispatch_ns`. A figure comparing the `routing_ns` bar across arms is comparing
shared code, so equality there is guaranteed by construction and is not
evidence that Algorithm 1 is cheap.

**`dispatch_ns` is a single bucket** holding the NVLink writes, the cross-rank
P2P barrier, Algorithm 1 and the receive-side filter, so a regression in any
one of them is indistinguishable from the others.

This commit widens the breakdown ring buffer from 9 to 27 int64 per slot
(65536 slots → 14 MB pinned) and appends the 18 adjacent deltas as a
`dar=[...]` field on each `Breakdown` line. The six existing categories are
computed exactly as before, so previously collected logs stay comparable. The
kernel-side write is guarded on `slot_stride_int64 >= 27`, so an older Python
side passing 9 still works.

`log_breakdown` runs *after* the dispatch and combine kernels and only writes
to the ring buffer, so it does not perturb the in-kernel timestamps the six
categories are derived from. This was verified by comparing category values
before and after the change.

**First result from it** (Qwen3 r64, M=32, 3 runs): the cross-rank barrier
(`fence2→p2p_wait`) is **0.07 µs**. The dispatch gap is not synchronisation
overhead. **15.8 µs of the ~17 µs gap is Algorithm 1's Pass 2.**

---

## 3. `perf:` drop the integer divide from METRO's serial greedy loop

Pass 2 of Algorithm 1 (`routing_mode=0`) walks the multi-replica experts on a
single thread, and for each candidate replica recomputed:

```c
r    = phys / epr;
slot = phys - r * epr;
```

sm_80 has no integer-divide instruction, so each `phys / epr` expands to a
software sequence — `I2F → MUFU.RCP → F2I → IMAD.WIDE` correction, plus `IABS`
for signs (disassembly shows 59 `MUFU.RCP` in the kernel). Inside parallel code
that is irrelevant; here it sits in an `if (threadIdx.x == 0)` block where no
other warp can hide the latency, and the loop runs it `rc` times per expert.

`epr` is uniform across the block, so the mapping is precomputed once in the
existing parallel preload into `s_l2p_rank[]` / `s_l2p_slot[]`
(`2 * NL * max_rep` int32, +2 KB shared, mode 0 only) and the greedy loop
indexes them. **The same values, computed by 256 threads instead of one.**

Measured over `rep` = 16/32/48/64 (Qwen3) and 8/16/24/32 (ERNIE), 2 runs each,
fitting Pass 2 time against redundant-expert count:

```
Qwen3   slope 0.1499 -> 0.0864 us/expert   (-42%)
ERNIE   slope 0.1913 -> 0.1116 us/expert   (-42%)
Pass 2 @ Qwen3 r64:   15.78 -> 10.30 us
Pass 2 @ ERNIE r32:    9.18 ->  6.18 us
preload cost:         +0.5 us  (the divides, now parallel)
```

The identical −42% on two models with different expert counts and hidden sizes
is the signature of removing a fixed per-iteration cost rather than changing
the algorithm.

**Routing decisions are unchanged, and this is checked rather than assumed.**
`expert_ns` reflects the activated-expert count directly and has a between-run
CV of 0.01%, so any altered decision would show up there. It moves by −0.01 µs
(Qwen3) and −0.07 µs (ERNIE); `combine_ns` and `gating_ns` likewise.

**Scope.** This is a constant-factor win of roughly 5 µs, about 1.3% of MoE-layer
time — likely below the noise floor of end-to-end TPOT (CV ≈ 1%). Pass 2 is
still `O(nm)` serial with a fixed floor, and at 2.0× replication on Qwen3 the
dispatch tax still cancels the expert-time saving.

### A rejected approach, recorded so it is not retried

The first attempt assumed the bottleneck was the `O(nm²)` insertion sort at the
top of Pass 2 and replaced the compact list with a direct `0..NL-1` scan. It was
**9.9 µs slower** (15.78 → 25.70) and was reverted.

The sort is not slow: Pass 1 fills `s_multi_experts` via `atomicAdd`, and
atomics to one address within a warp are hardware-serialised, so the array
arrives already segment-ordered and insertion sort degenerates to O(n). The
replacement widened the greedy loop from `nm≈64` to `NL=128` iterations, and
the extra no-op iterations cost more than the sort ever did.

A scaling experiment settled it afterwards: sweeping `rep` = 16/32/48/64 gives
*decreasing* marginal increments (+3.66, +2.58, +0.89 µs), so there is no
quadratic term. The lesson is that 19-slot resolution locates "Pass 2 is slow"
but cannot separate the sort from the greedy loop — that needed the scaling
measurement, not a guess.

---

## 4. `bench:` fix two measurement artifacts

Neither is a METRO or EPLB effect; both are properties of the harness that made
repeat runs disagree by more than the effect under test. Repeating one
configuration 3× per arm gave throughput CV 21–24%, p99 TTFT CV 147%, and p99
TPOT CV 0.5–1.1%. **Only TPOT was usable.**

### (a) The EPLB rearrange lands inside the measured window

`window_size = step_interval = 2 * NUM_PROMPTS` fires on a forward-step counter.
`bench_serve.sh` assumes warmup absorbs it, but nothing enforces that, and a
rearrange takes 13.3 s against a ~14 s measured window:

```
19:08:32  API server up
19:08:38  Rearranging experts ...
19:08:51  Rearranged experts in 13.28 seconds.
```

Two of six repeat runs were hit, showing 21.7 s duration against a 14.5 s norm.

`bench_serve_v2.sh` blocks after warmup until the log shows a completed
rearrange — the match string excludes the engine-startup `(profile)` pass, which
has a different message — then quiesces before measuring. It is a separate
script so that `bench_serve.sh` stays byte-comparable with the archived runs.

### (b) The prefill-heavy dataset is oversampled at every batch size

`vdaita/edit_5k_char` has 96 rows. The smallest configuration needs 128 requests
and bs=64 needs 512, so `maybe_oversample_requests` repeats rows 1.3×–5.3×.
Repeated prompts hit the prefix cache and skip prefill outright: at bs=64,
**63% of requests have TTFT = 0.0 ms** and the trimmed median is exactly 0. TTFT
there measures cache hit rate, not prefill. `vdaita/edit_10k_char` is not an
alternative — it has 90 rows.

`VLLM_SHAREGPT_LONGEST=1` makes `ShareGPTDataset` take the longest N prompts
instead of a random sample. ShareGPT has ~93k usable rows, so no batch size
oversamples, and the longest 512 are all ≥3000 tokens (against a median of 165
for the current prefill-heavy set). The default path is unchanged when the
variable is unset.

**Prefix caching is deliberately left on.** InstructCoder's 15–63% hit rate comes
from the shared chat template, which real serving has too, and it falls with
batch size (63.5% → 15.6%) as the shared prefix shrinks relative to total tokens.
Disabling caching would make the decode-heavy workload *less* realistic while
only masking the oversampling artifact.

---

## 5. `tools:` slurm harness and analysis scripts

All new files; no existing code is touched.

**`slurm/sweep_batch.slurm`** runs a list of configurations on one 8-GPU
allocation. Two hazards are handled explicitly:

- The bench scripts always write `results/<RUN_HASH>/`, so each configuration is
  cleared before the run and *moved* out after. Copying instead lets a second
  round silently overwrite the first.
- Resume must check the **artifact** matching the bench script, not directory
  existence: `bench_serve.sh` and `bench_breakdown.sh` share a `RUN_HASH`, so a
  directory check makes whichever runs second skip.

Also note: Della's `job_submit` plugin silently reroutes any job with a time
limit ≤ 1 h into the shared `gputest` partition, discarding `--partition`. All
scripts here request more than that.

**`tools/analysis/per_m_breakdown.py`** buckets by *exact* M. `M` is the MoE
layer's per-rank token count, padded to a CUDA-graph capture size, and METRO's
effect changes sign across that range — it loses at M=1 and wins at M≥16 — so
averaging over an M-range mixes regimes.

**`tools/analysis/repeat_variance.py`** reports mean/stdev/CV per metric across
repeated runs and labels each METRO-vs-EPLB delta significant or within-noise.

**`bench_serve_eager.sh`** is `bench_serve.sh` plus `--enforce-eager`. The
host-side DC profiler (`VLLM_DC_PROFILE`) synchronises and `cudaMemcpy`s from
Python, so it never fires under CUDA-graph replay; eager is the only way to
reach it. It is a copy rather than a flag because mutating `bench_serve.sh` in
place leaks into every queued job.

---

## Open items

- **Pass 2 scalability.** Still `O(nm)` serial. At 2.0× replication on Qwen3 the
  dispatch tax (+43.8 µs before this work) cancels the expert-time saving
  entirely. Options: parallelise the inner argmin (`rc` is usually 2, so little
  to gain); block-wise greedy trading a bounded quality loss for shorter serial
  depth; or overlap Pass 2 with attention — it depends only on the global expert
  counts, not on this rank's tokens, so it need not sit on the critical path.
- **TTFT and throughput are not yet reproducible.** Recomputing Figure 1 over
  three rounds gives TTFT sd 43.7 and throughput sd 40.0. TPOT reproduces well
  (ERNIE/InstructCoder 12.2 vs 12.2 in the paper; Qwen3/InstructCoder 8.2 vs
  8.4). The two fixes in §4 are prerequisites but have not been validated yet.
- **Figure 1 aggregation.** The caption says "Batch size 32", but taking bs=32
  reproduces only 5 of the 12 numbers while taking the max over batch sizes
  reproduces 12 of 12 — and the y-axis label already reads "max gain". The TTFT
  column is also mixed: three entries are the max, but ERNIE/Edit5kChar's 17.9%
  is the bs=64 value (its max is 58.3%).
