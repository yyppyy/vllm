#!/usr/bin/env python3
"""Before/after comparison for the Pass-2 integer-divide removal.

The only difference between the two sets is that commit "perf: drop the
integer divide from METRO's serial greedy loop" is applied. Breakdown
categories are in-kernel timestamp deltas and serve metrics come from
runs with the breakdown profiler off, so neither is perturbed by the
19-slot instrumentation that landed in between.
"""
import collections
import json
import math
import re
import statistics
import sys
from pathlib import Path

CATS = ["gating", "routing", "dispatch", "expert", "combine"]
DS = {0: "InstructCoder", 1: "Edit5kChar", 2: "ShareGPT"}


def brk(d, M=32):
    f = Path(d) / "server_breakdown.log"
    if not f.exists():
        return None
    acc = collections.defaultdict(list)
    for line in f.open():
        if not line.startswith("Breakdown"):
            continue
        kv = dict(re.findall(r"(\w+)=([-\d.]+)", line))
        if int(kv["M"]) != M:
            continue
        for c in CATS:
            if c + "_ns" in kv:
                acc[c].append(float(kv[c + "_ns"]) / 1000)
    return {c: statistics.mean(acc[c]) for c in CATS if acc[c]} or None


def serve(d):
    fs = sorted(list(Path(d).glob("bench_result_*.json"))
                + list(Path(d).glob("bench_result.json")))
    if not fs:
        return None
    tput = statistics.mean([json.load(open(f)).get("total_token_throughput", 0)
                            for f in fs])
    j = json.load(open(fs[-1]))
    itl, ol = j.get("itls"), j.get("output_lens")
    if itl and ol:
        tp = [sum(a) / (b - 1) * 1000 for a, b in zip(itl, ol) if b > 1 and a]
        k = max(1, math.ceil(0.01 * len(tp)))
        tpot = max(sorted(tp, reverse=True)[k:]) if len(tp) > k else None
    else:
        tpot = j.get("p99_tpot_ms")
    return {"tput": tput, "tpot": tpot}


def collect(roots, reader):
    """-> {config_key: [value_per_round]}"""
    out = collections.defaultdict(list)
    for r in roots:
        R = Path(r)
        if not R.exists():
            continue
        for d in R.iterdir():
            if not d.is_dir():
                continue
            t = d.name.split("_")
            if len(t) < 13 or t[2] != "1":
                continue
            v = reader(d)
            if v:
                out[("_".join(t[11:-1]), int(t[8]), int(t[4]), int(t[3]),
                     int(t[10]) > 0)].append(v)
    return out


def agg(vals, key):
    xs = [v[key] for v in vals if v.get(key) is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    pre_b = collect(["results/repro", "results/brk_r2", "results/brk_r3"], brk)
    post_b = collect([f"results/postdiv_brk_r{i}" for i in (1, 2, 3)], brk)
    pre_s = collect(["results/repro", "results/serve_r2", "results/serve_r3"], serve)
    post_s = collect([f"results/postdiv_serve_r{i}" for i in (1, 2)], serve)

    print("=" * 96)
    print("BREAKDOWN  M=32  MoE-only (us)   改前 = 除法优化前, 改后 = 优化后")
    print("=" * 96)
    print(f"{'model':14}{'ds':13}{'rep':>4}{'arm':7}"
          f"{'改前':>9}{'±sd':>7}{'改后':>9}{'±sd':>7}{'差':>8}{'%':>7}")
    print("-" * 96)
    for k in sorted(set(pre_b) & set(post_b)):
        model, ds, bs, rep, metro = k
        a = [sum(v.values()) for v in pre_b[k]]
        b = [sum(v.values()) for v in post_b[k]]
        ma, sa = statistics.mean(a), (statistics.stdev(a) if len(a) > 1 else 0)
        mb, sb = statistics.mean(b), (statistics.stdev(b) if len(b) > 1 else 0)
        print(f"{model[:14]:14}{DS.get(ds, ds):13}{rep:>4}"
              f"{'METRO' if metro else 'EPLB':7}"
              f"{ma:9.1f}{sa:7.2f}{mb:9.1f}{sb:7.2f}{mb-ma:+8.1f}{(mb-ma)/ma*100:+7.1f}")

    print()
    print("=" * 96)
    print("METRO vs EPLB 的 MoE gain%, 改前 vs 改后")
    print("=" * 96)
    print(f"{'model':14}{'ds':13}{'rep':>4}{'gain 改前':>11}{'gain 改后':>11}{'提升':>9}")
    print("-" * 96)
    for model, ds, bs, rep in sorted({(k[0], k[1], k[2], k[3]) for k in post_b if k[4]}):
        ke, km = (model, ds, bs, rep, False), (model, ds, bs, rep, True)
        rows = []
        for src in (pre_b, post_b):
            if ke not in src or km not in src:
                rows.append(None); continue
            e = statistics.mean([sum(v.values()) for v in src[ke]])
            m = statistics.mean([sum(v.values()) for v in src[km]])
            rows.append((e - m) / e * 100)
        if None in rows:
            continue
        print(f"{model[:14]:14}{DS.get(ds, ds):13}{rep:>4}"
              f"{rows[0]:11.1f}{rows[1]:11.1f}{rows[1]-rows[0]:+9.1f}")

    print()
    print("=" * 96)
    print("SERVE  端到端 (仅 METRO arm; TPOT 是 trim-max, throughput 是各 client 均值)")
    print("=" * 96)
    print(f"{'model':14}{'ds':13}{'bs':>4}{'rep':>4}"
          f"{'TPOT改前':>10}{'TPOT改后':>10}{'%':>7}"
          f"{'Tput改前':>10}{'Tput改后':>10}{'%':>7}")
    print("-" * 96)
    for k in sorted(set(pre_s) & set(post_s)):
        model, ds, bs, rep, metro = k
        if not metro:
            continue
        pa, _ = agg(pre_s[k], "tpot"); pb, _ = agg(post_s[k], "tpot")
        ta, _ = agg(pre_s[k], "tput"); tb, _ = agg(post_s[k], "tput")
        if None in (pa, pb, ta, tb):
            continue
        print(f"{model[:14]:14}{DS.get(ds, ds):13}{bs:>4}{rep:>4}"
              f"{pa:10.2f}{pb:10.2f}{(pa-pb)/pa*100:+7.1f}"
              f"{ta:10.0f}{tb:10.0f}{(tb-ta)/ta*100:+7.1f}")


if __name__ == "__main__":
    main()
