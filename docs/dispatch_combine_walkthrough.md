# dispatch_combine Backend: Code Walkthrough

Two data paths through the dispatch_combine all2all backend, depending on
whether `--mem-bound-aware-routing greedy` is specified.

---

## Path A: Normal dispatch_combine (no `--mem-bound-aware-routing`)

### Initialization

```
FusedMoE.__init__()
  └─ quant_method = UnquantizedFusedMoEMethod(moe)

prepare_communication_buffer_for_model()
  └─ quant_method.init_prepare_finalize(layer)
       └─ _maybe_make_prepare_finalize(moe, quant_config)
            └─ p2p_manager = all2all_manager.get_handle(...)  [dispatch_combine_buffers.py]
               DispatchCombinePrepareAndFinalize(
                   p2p_manager, ..., use_integrated_routing=False)

set_eplb_state(moe_layer_idx, expert_load_view, l2p_map, replica_count)
  └─ stores self.logical_to_physical_map, self.logical_replica_count
  └─ _maybe_init_integrated_routing()
       └─ checks moe_parallel_config.mem_bound_aware_routing
       └─ mem_bound_aware_routing is None → RETURNS EARLY, no-op
```

### Per-token forward (one MoE layer)

```
FusedMoE.forward()
  └─ quant_method.apply(enable_eplb=True, ...)
       └─ forward_cuda(layer, x, router_logits, enable_eplb=True, ...)

           ┌─────────────────────────────────────────────────────────┐
           │ 1. select_experts()                                     │
           │                                                         │
           │    _ir = False  (use_integrated_routing not set)        │
           │    eplb_for_select = True                               │
           │                                                         │
           │    topk_weights, topk_ids = fused_topk(router_logits)  │
           │      → topk_ids are LOGICAL expert IDs                  │
           │                                                         │
           │    eplb_map_to_physical_and_record(topk_ids, ...)       │
           │      → maps logical→physical IDs via l2p_map            │
           │      → records load into expert_load_view               │
           │      → topk_ids are now PHYSICAL expert IDs             │
           └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌─────────────────────────────────────────────────────────┐
           │ 2. fused_experts(topk_ids=PHYSICAL, ...)                │
           │    └─ prepare_async()                                   │
           │                                                         │
           │       use_integrated_routing = False                    │
           │       mc = M * topk  (capped at max_recv)               │
           │                                                         │
           │       ┌─ Kernel 1: dispatch_p2p ──────────────────────┐ │
           │       │  Each block = one (token, physical_expert) pair│ │
           │       │  dest_rank = expert_id / experts_per_rank      │ │
           │       │  atomicAdd dispatch_offset on dest_rank        │ │
           │       │  Write token data to dest's dispatch_recv      │ │
           │       │  Write metadata (rank, token_idx, expert_id,   │ │
           │       │                   topk_weight) to dispatch_meta│ │
           │       └───────────────────────────────────────────────┘ │
           │                                                         │
           │       ┌─ Kernel 2: prepare_dispatch_recv ─────────────┐ │
           │       │  Phase 1: Inline P2P barrier (RESET_COMBINE)  │ │
           │       │    block 0: reset combine_offset,              │ │
           │       │             threadfence_system,                 │ │
           │       │             flag exchange with all peers        │ │
           │       │    other blocks: spin on counter                │ │
           │       │                                                │ │
           │       │  Phase 2: Persistent loop over mc entries      │ │
           │       │    idx < actual: extract expert_id, weight     │ │
           │       │                  from metadata → output buffers│ │
           │       │                  atomicAdd expert_num_tokens   │ │
           │       │    idx >= actual: zero data, stamp sentinel    │ │
           │       └───────────────────────────────────────────────┘ │
           │                                                         │
           │    └─ _receiver()                                       │
           │       expert_x = dispatch_recv_tensor[:mc]              │
           │       quantize if needed                                │
           │       return (expert_x, expert_topk_ids,                │
           │               expert_topk_weights, expert_num_tokens)   │
           └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌─────────────────────────────────────────────────────────┐
           │ 3. Expert computation (TritonExperts)                   │
           │    For each local expert:                               │
           │      gate_up = expert_x @ w13   (fused gate+up proj)   │
           │      act = silu(gate) * up                              │
           │      down = act @ w2            (down projection)       │
           └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌─────────────────────────────────────────────────────────┐
           │ 4. finalize()                                           │
           │                                                         │
           │    weight_and_reduce: apply topk weights to expert out  │
           │                                                         │
           │    ┌─ Kernel 3: combine_p2p ──────────────────────────┐ │
           │    │  Each block = one dispatched entry                │ │
           │    │  Read source_rank, source_token_idx from metadata │ │
           │    │  Write expert output back to source rank's        │ │
           │    │  combine_recv buffer at atomicAdd'd offset        │ │
           │    │                                                   │ │
           │    │  (weight==0 entries are SKIPPED — combine         │ │
           │    │   optimization for filtered tokens)               │ │
           │    └──────────────────────────────────────────────────┘ │
           │                                                         │
           │    ┌─ Kernel 4: p2p_barrier (RESET_DISPATCH) ────────┐ │
           │    │  Sync combine writes across all ranks            │ │
           │    │  Reset dispatch_offset to 0 for next layer       │ │
           │    └──────────────────────────────────────────────────┘ │
           │                                                         │
           │    ┌─ Kernel 5: scatter_add_direct ───────────────────┐ │
           │    │  cudaMemsetAsync zeros output                    │ │
           │    │  Read combine_recv via IPC pointers               │ │
           │    │  atomicAdd each token's result back into          │ │
           │    │  output[source_token_idx]                         │ │
           │    └──────────────────────────────────────────────────┘ │
           └─────────────────────────────────────────────────────────┘
```

### Key properties
- **Routing decision**: Made locally per-rank BEFORE dispatch, using
  `eplb_map_to_physical_and_record()`. Each rank picks a physical replica
  based only on its own topk selections (locally-optimal, no cross-rank
  demand visibility).
- **Dispatch target**: Each (token, expert) pair sent to exactly ONE rank
  (the rank that owns the selected physical expert).
- **Kernel count**: 5 kernels per MoE layer (dispatch, prepare, expert,
  combine, barrier+scatter).
- **Barrier count**: 2 P2P barriers (post-dispatch RESET_COMBINE,
  post-combine RESET_DISPATCH).

---

## Path B: Integrated routing (`--mem-bound-aware-routing greedy`)

### Initialization

```
FusedMoE.__init__()
  └─ (same as Path A)

prepare_communication_buffer_for_model()
  └─ (same as Path A, use_integrated_routing=False initially)

set_eplb_state(moe_layer_idx, expert_load_view, l2p_map, replica_count)
  └─ stores self.logical_to_physical_map, self.logical_replica_count
  └─ _maybe_init_integrated_routing()
       └─ checks moe_parallel_config.mem_bound_aware_routing
       └─ mem_bound_aware_routing = "greedy" → PROCEEDS
       └─ pf = DispatchCombinePrepareAndFinalize (from quant_method)
       └─ mgr = pf.p2p_manager
       └─ mgr.init_integrated_routing(NL, max_replicas, epr)
            ├─ cudaMalloc expert_counts buffer (NL * 4 bytes)
            ├─ Exchange IPC handles for expert_counts
            │  (all ranks can atomicAdd to each other's buffer)
            ├─ cudaMalloc routing_selection buffer (NL * 4 bytes)
            ├─ cudaMalloc routing_ready_flag (4 bytes)
            ├─ Allocate routing_map_tensor, routing_count_tensor (GPU)
            ├─ Wrap expert_counts as tensor for CUDA graph compat
            └─ Rebuild config_tensor with new fields
       └─ mgr.update_routing_tables(l2p_map, replica_count)
            ├─ Copy l2p_map → routing_map_tensor (GPU)
            └─ Copy replica_count → routing_count_tensor (GPU)
       └─ pf.use_integrated_routing = True
       └─ pf.expert_load_view = expert_load_view
```

### Per-token forward (one MoE layer)

```
FusedMoE.forward()
  └─ quant_method.apply(enable_eplb=True, ...)
       └─ forward_cuda(layer, x, router_logits, enable_eplb=True, ...)

           ┌─────────────────────────────────────────────────────────┐
           │ 1. select_experts()                                     │
           │                                                         │
           │    _ir = True  (use_integrated_routing is set)          │
           │    eplb_for_select = False                              │
           │                                                         │
           │    topk_weights, topk_ids = fused_topk(router_logits)  │
           │      → topk_ids are LOGICAL expert IDs                  │
           │                                                         │
           │    eplb_map_to_physical_and_record() is SKIPPED         │
           │      → topk_ids STAY as LOGICAL expert IDs              │
           │      → no load recording here (done later in kernel)    │
           └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌─────────────────────────────────────────────────────────┐
           │ 2. fused_experts(topk_ids=LOGICAL, ...)                 │
           │    └─ prepare_async()                                   │
           │                                                         │
           │       use_integrated_routing = True                     │
           │       → _prepare_integrated()                           │
           │       mc = M * topk * max_replicas  (expanded volume)   │
           │                                                         │
           │   ┌─ ONE FUSED KERNEL: dispatch_and_route_kernel ─────┐ │
           │   │  Persistent grid (32 blocks × 256 threads)         │ │
           │   │                                                    │ │
           │   │  ── Phase A: Broadcast dispatch + push all-reduce ─│ │
           │   │  Loop over (token, expert_slot, replica) triples:  │ │
           │   │    logical_id = topk_ids[token * topk + slot]      │ │
           │   │    For EACH replica of logical_id:                  │ │
           │   │      phys_id = l2p_map[logical_id * max_rep + r]   │ │
           │   │      dest_rank = phys_id / experts_per_rank        │ │
           │   │      atomicAdd dest's dispatch_offset               │ │
           │   │      Write token data → dest's dispatch_recv       │ │
           │   │      Write metadata (LOGICAL expert_id)             │ │
           │   │                                                    │ │
           │   │    Push all-reduce (once per token,slot):           │ │
           │   │      for each rank r:                               │ │
           │   │        atomicAdd(&remote_expert_counts[r]           │ │
           │   │                   [logical_id], 1)                  │ │
           │   │      → After barrier, each rank's local buffer      │ │
           │   │        has GLOBAL per-expert token counts           │ │
           │   │                                                    │ │
           │   │  ── Phase B: P2P barrier (RESET_COMBINE) ─────────│ │
           │   │    One barrier covers BOTH:                         │ │
           │   │      - dispatch data writes (token payloads)        │ │
           │   │      - remote expert_counts atomicAdds              │ │
           │   │    block 0: reset combine_offset,                   │ │
           │   │             threadfence_system,                      │ │
           │   │             flag exchange with all peers             │ │
           │   │    other blocks: spin on counter                    │ │
           │   │                                                    │ │
           │   │  ── Phase C: Deterministic router (block 0) ──────│ │
           │   │    Read global counts from local expert_counts buf │ │
           │   │    For each logical expert e (ascending order):     │ │
           │   │      If single replica: select it                   │ │
           │   │      If multiple replicas: pick the replica whose   │ │
           │   │        rank has the lowest active token count       │ │
           │   │        (ties broken by lower rank)                  │ │
           │   │      routing_selection[e] = chosen physical_id      │ │
           │   │      rank_active_counts[chosen_rank] += count       │ │
           │   │    Write routing_selection to global memory          │ │
           │   │    Signal routing_ready_flag (monotonic counter)     │ │
           │   │                                                    │ │
           │   │    Other blocks: spin on routing_ready_flag         │ │
           │   │                                                    │ │
           │   │  ── Phase D: Filter + stamp/zero + metadata ──────│ │
           │   │    Persistent loop over mc entries:                  │ │
           │   │    idx < actual_dispatched:                          │ │
           │   │      logical_id = metadata[idx].expert_id           │ │
           │   │      selected_phys = routing_selection[logical_id]   │ │
           │   │      selected_rank = selected_phys / epr            │ │
           │   │      if selected_rank == my_rank:                   │ │
           │   │        KEEP: set expert_topk_ids = selected_phys    │ │
           │   │              set expert_topk_weights = meta.weight   │ │
           │   │              atomicAdd expert_num_tokens             │ │
           │   │      else:                                          │ │
           │   │        FILTER: expert_topk_ids = sentinel           │ │
           │   │                expert_topk_weights = 0              │ │
           │   │                metadata.topk_weight = 0             │ │
           │   │                (combine_p2p will skip weight==0)     │ │
           │   │    idx >= actual_dispatched:                         │ │
           │   │      Zero data, stamp sentinel metadata              │ │
           │   └───────────────────────────────────────────────────┘ │
           │                                                         │
           │    Record load: expert_load_view += expert_num_tokens   │
           │      (for EPLB rebalancing decisions)                   │
           │                                                         │
           │    └─ _receiver()                                       │
           │       (same as Path A: slice, quantize, return)         │
           └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌─────────────────────────────────────────────────────────┐
           │ 3. Expert computation (TritonExperts)                   │
           │    (same as Path A, but ~50% of entries are filtered    │
           │     with sentinel expert_id → expert skips them)        │
           └─────────────────────────────────────────────────────────┘
                                    │
                                    ▼
           ┌─────────────────────────────────────────────────────────┐
           │ 4. finalize()                                           │
           │    (same as Path A)                                     │
           │                                                         │
           │    combine_p2p: weight==0 entries are SKIPPED           │
           │    (filtered tokens have weight=0 in metadata,          │
           │     so combine traffic is reduced by ~50%)              │
           └─────────────────────────────────────────────────────────┘
```

### Key properties
- **Routing decision**: Made AFTER dispatch with GLOBAL demand visibility.
  All ranks run the identical deterministic router on the same global
  per-expert token counts, producing identical `routing_selection`. This
  enables globally-optimal load balancing.
- **Dispatch target**: Each (token, expert) pair broadcast to ALL ranks
  that hold a replica of that expert. Filtered after routing.
- **Kernel count**: Same 5 logical phases but dispatch+barrier+route+filter
  are fused into ONE kernel launch (`dispatch_and_route_kernel`).
  Total: 4 kernel launches per MoE layer (fused dispatch+route, expert,
  combine, barrier+scatter).
- **Barrier count**: Same 2 P2P barriers, but the first is inside the
  fused kernel, shared between dispatch data and all-reduce counts.
- **All-reduce mechanism**: Push-based remote atomicAdd to all peers' IPC
  expert_counts buffers during dispatch (no separate all-reduce phase).
  After barrier, each rank's local buffer has the global sum.
- **CUDA graph compatible**: No host-side GPU pointer dereferences.
  `routing_ready_flag` uses monotonic counter pattern (same as P2P barrier).
  All `cudaMemsetAsync` targets are passed as tensor arguments.

---

## Comparison

| Aspect | Path A (normal) | Path B (mem-bound-aware) |
|--------|-----------------|--------------------------|
| Routing location | Pre-dispatch (host-side) | Post-dispatch (GPU-side, fused) |
| Routing visibility | Local rank only | Global (all ranks' demand) |
| Dispatch volume | 1× (one dest per pair) | max_replicas× (broadcast) |
| Combine volume | 1× | ~0.5× (filtered entries skipped) |
| Kernel launches | 5 per layer | 4 per layer (fused dispatch+route) |
| P2P barriers | 2 | 2 (one shared for dispatch+allreduce) |
| Load balance quality | Locally optimal | Globally optimal |
| Extra IPC buffers | None | expert_counts (NL×4B per rank) |
| Activation flag | Default | `--mem-bound-aware-routing greedy` |

---

## File Map

| File | Role |
|------|------|
| `csrc/dispatch_combine.cuh` | Config struct, all CUDA kernels (dispatch, combine, barrier, fused dispatch+route) |
| `csrc/dispatch_combine.cu` | Host wrappers for all kernels |
| `csrc/torch_bindings.cpp` | PyTorch op registration (`_C_dispatch_combine` library) |
| `vllm/distributed/device_communicators/dispatch_combine_buffers.py` | IPC buffer manager, config tensor packing, GPU-side op wrappers |
| `vllm/model_executor/layers/fused_moe/dispatch_combine_prepare_finalize.py` | Python orchestration: prepare (dispatch) and finalize (combine) |
| `vllm/model_executor/layers/fused_moe/layer.py` | FusedMoE layer: routing, EPLB integration, forward dispatch |
