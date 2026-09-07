#!/usr/bin/env python3
"""Run-to-run variance for the bench_serve metrics.

Every gate in the optimization plan is stated as a +/-X% threshold, which only
means something if run-to-run spread is smaller than X. This reports the spread
per arm and the METRO-vs-EPLB delta with its uncertainty.

Usage: python3 tools/analysis/repeat_variance.py [results/repeat]
"""
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/repeat")
METRICS = [("total_token_throughput", "throughput", "tok/s", True),
           ("p99_tpot_ms", "p99 TPOT", "ms", False),
           ("p99_ttft_ms", "p99 TTFT", "ms", False),
           ("p99_e2el_ms", "p99 E2EL", "ms", False)]

arms = {}
for d in sorted(root.glob("*_run*")):
    arm = d.name.rsplit("_run", 1)[0]
    vals = {}
    for f in sorted(list(d.glob("bench_result_*.json")) + list(d.glob("bench_result.json"))):
        j = json.load(open(f))
        for key, _, _, additive in METRICS:
            v = j.get(key)
            if v is None:
                continue
            vals[key] = vals.get(key, 0) + v if additive else max(vals.get(key, 0), v)
    if vals:
        arms.setdefault(arm, []).append(vals)

if not arms:
    sys.exit(f"no runs found under {root}")

print(f"{'metric':12}{'arm':8}{'n':>3}{'mean':>11}{'stdev':>10}{'cv%':>7}{'min':>10}{'max':>10}")
print("-" * 71)
summary = {}
for key, name, unit, _ in METRICS:
    for arm in sorted(arms):
        xs = [v[key] for v in arms[arm] if key in v]
        if not xs:
            continue
        mu = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        summary[(key, arm)] = (mu, sd, len(xs))
        print(f"{name:12}{arm:8}{len(xs):>3}{mu:11.2f}{sd:10.2f}"
              f"{100*sd/mu if mu else 0:7.1f}{min(xs):10.2f}{max(xs):10.2f}")
    print()

print("METRO vs EPLB  (+ = METRO better; +/- is propagated stdev)")
print("-" * 71)
for key, name, unit, higher_better in METRICS:
    a, b = summary.get((key, "EPLB")), summary.get((key, "METRO"))
    if not a or not b:
        continue
    (ma, sa, na), (mb, sb, nb) = a, b
    delta = (mb - ma) / ma * 100 if higher_better else (ma - mb) / ma * 100
    # relative-difference error propagation
    err = 100 * ((sb / ma) ** 2 + (mb * sa / ma ** 2) ** 2) ** 0.5
    sig = "significant" if abs(delta) > 2 * err else "WITHIN NOISE"
    print(f"  {name:12}{delta:+7.1f}%  +/- {err:4.1f}%   {sig}")
