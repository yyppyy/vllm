<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

---
Join us at the [PyTorch Conference, October 22-23](https://events.linuxfoundation.org/pytorch-conference/) and [Ray Summit, November 3-5](https://www.anyscale.com/ray-summit/2025) in San Francisco for our latest updates on vLLM and to meet the vLLM team! Register now for the largest vLLM community events of the year!

---

*Latest News* 🔥

- [2025/08] We hosted [vLLM Shenzhen Meetup](https://mp.weixin.qq.com/s/k8ZBO1u2_2odgiKWH_GVTQ) focusing on the ecosystem around vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Ua2SVKVSu-wp5vou_6ElraDt2bnKhiEA).
- [2025/08] We hosted [vLLM Singapore Meetup](https://www.sginnovate.com/event/vllm-sg-meet). We shared V1 updates, disaggregated serving and MLLM speedups with speakers from Embedded LLM, AMD, WekaIO, and A*STAR. Please find the meetup slides [here](https://drive.google.com/drive/folders/1ncf3GyqLdqFaB6IeB834E5TZJPLAOiXZ?usp=sharing).
- [2025/08] We hosted [vLLM Shanghai Meetup](https://mp.weixin.qq.com/s/pDmAXHcN7Iqc8sUKgJgGtg) focusing on building, developing, and integrating with vLLM! Please find the meetup slides [here](https://drive.google.com/drive/folders/1OvLx39wnCGy_WKq8SiVKf7YcxxYI3WCH).
- [2025/05] vLLM is now a hosted project under PyTorch Foundation! Please find the announcement [here](https://pytorch.org/blog/pytorch-foundation-welcomes-vllm/).
- [2025/01] We are excited to announce the alpha release of vLLM V1: A major architectural upgrade with 1.7x speedup! Clean code, optimized execution loop, zero-overhead prefix caching, enhanced multimodal support, and more. Please check out our blog post [here](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html).

<details>
<summary>Previous News</summary>

- [2025/08] We hosted [vLLM Korea Meetup](https://luma.com/cgcgprmh) with Red Hat and Rebellions! We shared the latest advancements in vLLM along with project spotlights from the vLLM Korea community. Please find the meetup slides [here](https://drive.google.com/file/d/1bcrrAE1rxUgx0mjIeOWT6hNe2RefC5Hm/view).
- [2025/08] We hosted [vLLM Beijing Meetup](https://mp.weixin.qq.com/s/dgkWg1WFpWGO2jCdTqQHxA) focusing on large-scale LLM deployment! Please find the meetup slides [here](https://drive.google.com/drive/folders/1Pid6NSFLU43DZRi0EaTcPgXsAzDvbBqF) and the recording [here](https://www.chaspark.com/#/live/1166916873711665152).
- [2025/05] We hosted [NYC vLLM Meetup](https://lu.ma/c1rqyf1f)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1_q_aW_ioMJWUImf1s1YM-ZhjXz8cUeL0IJvaquOYBeA/edit?usp=sharing).
- [2025/04] We hosted [Asia Developer Day](https://www.sginnovate.com/event/limited-availability-morning-evening-slots-remaining-inaugural-vllm-asia-developer-day)! Please find the meetup slides from the vLLM team [here](https://docs.google.com/presentation/d/19cp6Qu8u48ihB91A064XfaXruNYiBOUKrBxAmDOllOo/edit?usp=sharing).
- [2025/03] We hosted [vLLM x Ollama Inference Night](https://lu.ma/vllm-ollama)! Please find the meetup slides from the vLLM team [here](https://docs.google.com/presentation/d/16T2PDD1YwRnZ4Tu8Q5r6n53c5Lr5c73UV9Vd2_eBo4U/edit?usp=sharing).
- [2025/03] We hosted [the first vLLM China Meetup](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg)! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1REHvfQMKGnvz6p3Fd23HhSO4c8j5WPGZV0bKYLwnHyQ/edit?usp=sharing).
- [2025/03] We hosted [the East Coast vLLM Meetup](https://lu.ma/7mu4k4xx)! Please find the meetup slides [here](https://docs.google.com/presentation/d/1NHiv8EUFF1NLd3fEYODm56nDmL26lEeXCaDgyDlTsRs/edit#slide=id.g31441846c39_0_0).
- [2025/02] We hosted [the ninth vLLM meetup](https://lu.ma/h7g3kuj9) with Meta! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1jzC_PZVXrVNSFVCW-V4cFXb6pn7zZ2CyP_Flwo05aqg/edit?usp=sharing) and AMD [here](https://drive.google.com/file/d/1Zk5qEJIkTmlQ2eQcXQZlljAx3m9s7nwn/view?usp=sharing). The slides from Meta will not be posted.
- [2025/01] We hosted [the eighth vLLM meetup](https://lu.ma/zep56hui) with Google Cloud! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1epVkt4Zu8Jz_S5OhEHPc798emsYh2BwYfRuDDVEF7u4/edit?usp=sharing), and Google Cloud team [here](https://drive.google.com/file/d/1h24pHewANyRL11xy5dXUbvRC9F9Kkjix/view?usp=sharing).
- [2024/12] vLLM joins [pytorch ecosystem](https://pytorch.org/blog/vllm-joins-pytorch)! Easy, Fast, and Cheap LLM Serving for Everyone!
- [2024/11] We hosted [the seventh vLLM meetup](https://lu.ma/h0qvrajz) with Snowflake! Please find the meetup slides from vLLM team [here](https://docs.google.com/presentation/d/1e3CxQBV3JsfGp30SwyvS3eM_tW-ghOhJ9PAJGK6KR54/edit?usp=sharing), and Snowflake team [here](https://docs.google.com/presentation/d/1qF3RkDAbOULwz9WK5TOltt2fE9t6uIc_hVNLFAaQX6A/edit?usp=sharing).
- [2024/10] We have just created a developer slack ([slack.vllm.ai](https://slack.vllm.ai)) focusing on coordinating contributions and discussing features. Please feel free to join us there!
- [2024/10] Ray Summit 2024 held a special track for vLLM! Please find the opening talk slides from the vLLM team [here](https://docs.google.com/presentation/d/1B_KQxpHBTRa_mDF-tR6i8rWdOU5QoTZNcEg2MKZxEHM/edit?usp=sharing). Learn more from the [talks](https://www.youtube.com/playlist?list=PLzTswPQNepXl6AQwifuwUImLPFRVpksjR) from other vLLM contributors and users!
- [2024/09] We hosted [the sixth vLLM meetup](https://lu.ma/87q3nvnh) with NVIDIA! Please find the meetup slides [here](https://docs.google.com/presentation/d/1wrLGwytQfaOTd5wCGSPNhoaW3nq0E-9wqyP7ny93xRs/edit?usp=sharing).
- [2024/07] We hosted [the fifth vLLM meetup](https://lu.ma/lp0gyjqr) with AWS! Please find the meetup slides [here](https://docs.google.com/presentation/d/1RgUD8aCfcHocghoP3zmXzck9vX3RCI9yfUAB2Bbcl4Y/edit?usp=sharing).
- [2024/07] In partnership with Meta, vLLM officially supports Llama 3.1 with FP8 quantization and pipeline parallelism! Please check out our blog post [here](https://blog.vllm.ai/2024/07/23/llama31.html).
- [2024/06] We hosted [the fourth vLLM meetup](https://lu.ma/agivllm) with Cloudflare and BentoML! Please find the meetup slides [here](https://docs.google.com/presentation/d/1iJ8o7V2bQEi0BFEljLTwc5G1S10_Rhv3beed5oB0NJ4/edit?usp=sharing).
- [2024/04] We hosted [the third vLLM meetup](https://robloxandvllmmeetup2024.splashthat.com/) with Roblox! Please find the meetup slides [here](https://docs.google.com/presentation/d/1A--47JAK4BJ39t954HyTkvtfwn0fkqtsL8NGFuslReM/edit?usp=sharing).
- [2024/01] We hosted [the second vLLM meetup](https://lu.ma/ygxbpzhl) with IBM! Please find the meetup slides [here](https://docs.google.com/presentation/d/12mI2sKABnUw5RBWXDYY-HtHth4iMSNcEoQ10jDQbxgA/edit?usp=sharing).
- [2023/10] We hosted [the first vLLM meetup](https://lu.ma/first-vllm-meetup) with a16z! Please find the meetup slides [here](https://docs.google.com/presentation/d/1QL-XPFXiFpDBh86DbEegFXBXFXjix4v032GhShbKf3s/edit?usp=sharing).
- [2023/08] We would like to express our sincere gratitude to [Andreessen Horowitz](https://a16z.com/2023/08/30/supporting-the-open-source-ai-community/) (a16z) for providing a generous grant to support the open-source development and research of vLLM.
- [2023/06] We officially released vLLM! FastChat-vLLM integration has powered [LMSYS Vicuna and Chatbot Arena](https://chat.lmsys.org) since mid-April. Check out our [blog post](https://vllm.ai).

</details>

---

## About

- first-time setup: ```./setup.sh```
- incremental setup: ```./setup.sh --incremental```
- run end-to-end benchmark: ```./bench_serve.sh 8 8 0 32 0 0 likaixin/InstructCoder```; copy ```bench_result_*.json``` from GPU server to ```results```
- run latency breakdown: ```git checkout latency_breakdown```; uncomment redundant components (search for redundant) in ```vllm/model_executor/layers/fused_moe/fused_moe.py```; run end-to-end, subtract original end-to-end to get the redundat component time. enter them into ```latency_breakdown.csv```. Note that results derived from ```server_*.log```.
- copy results from the analytical framework into ```routing_solver.csv```

- [incremental compilation workflow](https://docs.vllm.ai/en/stable/contributing/incremental_build.html#prerequisites)
- [vllm bench serve](https://docs.vllm.ai/en/latest/cli/bench/serve.html#options)
- [EPLB routing](https://github.com/vllm-project/vllm/blob/8bf8f4582208ac7af230512ff5f3ac1dc36d5222/vllm/model_executor/layers/fused_moe/fused_moe.py#L1110)
- [supported models](https://docs.vllm.ai/en/v0.10.2/models/supported_models.html)
- [dataset:humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Benchmarking & Profiling Pipelines (research fork)

This fork ships four bench scripts at the repo root and a matching set of
plotting scripts under `plots/`. Each bench script launches a
`vllm serve` engine, drives a load against it, drops one or more
result files into `results/vllm_results_final/<RUN_HASH>/`, and shuts
the engine down cleanly. The plotting scripts then sweep that
directory and emit PDFs into `plots/`. The four scripts share a
single 13-positional-arg signature so the same invocation can be
re-run under a different profiler simply by swapping the script name.

### One-time setup

```bash
./setup.sh                       # build C++/CUDA extensions, create venv
./download_models.sh             # populates ./models/<MODEL_NAME>/
./download_datasets.sh           # populates ./datasets/
```

### Common positional arguments

All four bench scripts accept the same 13 positional args (the 10th —
`USE_PROFILER` — is honored only by `bench_serve.sh`; the other three
ignore it for signature parity):

| # | Name | Meaning |
| - | ---- | ------- |
| 1 | `NUM_GPUS` | total GPU count visible to the engine |
| 2 | `EP_DEGREE` | data-parallel size (= EP world size when `USE_EP=1`) |
| 3 | `USE_EP` | 1 = expert parallel via dispatch_combine; 0 = TP via allgather |
| 4 | `NUM_REPLICAS` | EPLB `num_redundant_experts` |
| 5 | `BATCH_SIZE` | per-GPU `--max-num-seqs`; total prompts = `BATCH_SIZE × NUM_GPUS` |
| 6 | `MEM_BOUND_ROUTING` | 0=off, 1/2/3=routing-mode variants (sets `VLLM_PREFILL_ROUTING_MODE`) |
| 7 | `ALLTOALL_BACKEND` | `dispatch_combine` or `allgather_reducescatter` |
| 8 | `DATASET` | numeric id baked into the run hash (0=InstructCoder, 1=Edit_5k_char, 2=ShareGPT) |
| 9 | `DATASET_NAME` | actual dataset selector — `random`, `sharegpt`, or an HF dataset path |
| 10 | `USE_PROFILER` | nsys gate — only `bench_serve.sh` uses it; others ignore |
| 11 | `MEM_BOUND_ROUTING_THRES` | METRO discriminator: `>0` = vllm-METRO, `0` = vllm-EP |
| 12 | `MODEL_NAME` | dir name under `./models/` |
| 13 | `EPLB_NUM_GROUPS` | EPLB grouping factor (default 1) |

The `RUN_HASH` is `${1}_${2}_${3}_${4}_${5}_${6}_${7}_${8}_${10}_${11}_${12}_g${13}`.

### `bench_serve.sh` → throughput-vs-latency

The main throughput / TTFT / TPOT bench. Multi-client closed-loop
sweep (≥128 prompts total) with `bench_result_<idx>.json` per client.
Optionally wraps `vllm serve` in `nsys profile` when `USE_PROFILER>0`.

```bash
# Qwen3 30B / 8 GPUs / EP / 64 redundant experts / batch 32 / METRO routing
./bench_serve.sh 8 8 1 64 32 2 dispatch_combine 0 likaixin/InstructCoder \
                 0 256 Qwen3-30B-A3B-8-128 1
```

Outputs:

- `results/vllm_results_final/<RUN_HASH>/bench_result_*.json`
- `results/vllm_results_final/<RUN_HASH>/server.log` (engine stdout)
- `results/vllm_results_final/<RUN_HASH>/profile.nsys-rep` (only if `USE_PROFILER=1`)

Plot:

```bash
python3 plots/plot_throughput_latency.py
# -> plots/throughput_vs_p99{tpot,ttft}_<model>_<dataset>.pdf
```

`plot_throughput_latency.py` discriminates four series per `(model, dataset)`
figure: TP, vllm-EP 1.0x, vllm-EP 1.5x, vllm-METRO 1.5x — derived from
the `(USE_EP, NUM_REPLICAS, ALLTOALL_BACKEND, MEM_BOUND_ROUTING_THRES)`
tuple parsed out of `RUN_HASH`.

### `bench_exp_vs_latency.sh` → per-kernel ExpLat profile

Same engine config as `bench_serve.sh` but with
`VLLM_EXP_LATENCY_PROFILE=1` and a sentinel-file gate so the warmup +
graph-capture passes produce no records. The in-process poller dumps
the pinned ringbuffer to disk 35 s after the bench finishes.

```bash
./bench_exp_vs_latency.sh 8 8 1 64 32 2 dispatch_combine 0 \
                          likaixin/InstructCoder 0 256 \
                          Qwen3-30B-A3B-8-128 1
```

Outputs:

- `results/vllm_results_final/<RUN_HASH>/server_explat.log` — one
  `ExpLat seq=… rank=… layer=… M=… num_local_experts=… align_ns=…
  gemm_gu_ns=… silu_ns=… quant_ns=… gemm_dn_ns=…
  per_expert_tokens=[…]` line per (rank, layer, batch) replay.
- `results/vllm_results_final/<RUN_HASH>/server_main.log`

Plot:

```bash
python3 plots/plot_activated_vs_latency.py
# -> plots/activated_vs_latency_exp_xlt_box_<model>_<dataset>.pdf
# -> plots/activated_vs_latency_exp_xnact_box_<model>_<dataset>.pdf
```

`plot_activated_vs_latency.py` scatters number-of-activated-experts
against MoE-kernel latency, one figure per `(model, dataset)`.

### `bench_tok_cnt.sh` → per-expert token-count profile

Reuses the ExpLat infrastructure but flips `VLLM_ZIPFIAN_ROUTING=0`
and uses a Poisson-arrival client (Little's law steady state). Output
log has the same format as `server_explat.log` — the parser focuses
on the `per_expert_tokens` field for the memory-bound CDF.

```bash
./bench_tok_cnt.sh 8 8 1 64 32 2 dispatch_combine 0 \
                   likaixin/InstructCoder 0 256 \
                   Qwen3-30B-A3B-8-128 1
```

Outputs:

- `results/vllm_results_final/<RUN_HASH>/server_tokcnt.log` (or
  `.log.gz` for runs that exceed GitHub's 100 MB push limit — the
  plotting script reads either)

Plot:

```bash
python3 plots/plot_membound_cdf.py --gpu A100_40GB
# -> plots/membound_cdf_<model>_<dataset>.pdf
```

For each `(model, dataset)` the script computes a per-record
"fraction of local experts that are memory-bound" using a roofline
threshold derived from the model's HF config and the `--gpu` preset,
then draws one CDF line per bench batch-size config. Pass
`--block-m 0 --block-n 0` to disable the SRAM-tile cap and use the
GPU compute ridge alone.

### `bench_breakdown.sh` → 6-category latency breakdown

Captureable %globaltimer reads inside `dispatch_and_route_kernel` /
`combine_and_scatter_kernel` plus python-side `record_stamp` ops at
the attention / MoE boundaries. The single closed-loop client matches
`bench_serve.sh` (no Poisson, no looping). Disables the
torch.compile cache (`VLLM_DISABLE_COMPILE_CACHE=1`) since the
breakdown profiler lifts extra tensor attrs into the FX graph and a
stale cache from a non-breakdown run would crash inductor.

```bash
./bench_breakdown.sh 8 8 1 64 32 2 dispatch_combine 0 \
                     likaixin/InstructCoder 0 256 \
                     Qwen3-30B-A3B-8-128 1
```

Outputs:

- `results/vllm_results_final/<RUN_HASH>/server_breakdown.log` —
  one `Breakdown seq=… rank=… layer=… M=…
  attention_ns=… gating_ns=… routing_ns=… dispatch_ns=…
  expert_ns=… combine_ns=…` line per (rank, layer, batch).
- `results/vllm_results_final/<RUN_HASH>/server_main.log`

Plot:

```bash
python3 plots/plot_latency_breakdown_M.py            # default --x-max 700, --m-range 24-32
# -> plots/latency_breakdown_M_<model>_<dataset>.pdf
```

`plot_latency_breakdown_M.py` averages each of the six categories
across every record whose per-replay `M` falls in `--m-range`
(default `24-32`), buckets by replication ratio
`(num_experts + NUM_REPLICAS) / num_experts`, and draws horizontal
stacked bars with vllm-EP and vllm-METRO side by side. Per-figure
x-axis caps are kept in `X_MAX_OVERRIDES` at the top of the script.

### Sweep helpers

`bench_serve_8GPU.sh` and `log_bench_serve.sh` are loop wrappers that
invoke `bench_serve.sh` over a parameter sweep (replication ratio,
batch size, dataset). `style.py` under `plots/` defines the unified
publication-style palette, font, and 3.2 × 2.4 in panel size that all
plotting scripts inherit via `apply_style()` + `paper_figure()`.

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Sponsors

vLLM is a community project. Our compute resources for development and testing are supported by the following organizations. Thank you for your support!

<!-- Note: Please sort them in alphabetical order. -->
<!-- Note: Please keep these consistent with docs/community/sponsors.md -->
Cash Donations:

- a16z
- Dropbox
- Sequoia Capital
- Skywork AI
- ZhenFund

Compute Resources:

- Alibaba Cloud
- AMD
- Anyscale
- AWS
- Crusoe Cloud
- Databricks
- DeepInfra
- Google Cloud
- Intel
- Lambda Lab
- Nebius
- Novita AI
- NVIDIA
- Replicate
- Roblox
- RunPod
- Trainy
- UC Berkeley
- UC San Diego

Slack Sponsor: Anyscale

We also have an official fundraising venue through [OpenCollective](https://opencollective.com/vllm). We plan to use the fund to support the development, maintenance, and adoption of vLLM.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [vllm-questions@lists.berkeley.edu](mailto:vllm-questions@lists.berkeley.edu)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
