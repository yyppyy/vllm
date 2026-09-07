#!/usr/bin/env python3
"""Per-M MoE latency breakdown, METRO vs EPLB.

`M` in server_breakdown.log is the CUDA-graph padded size, not the raw token
count, so aggregating over an M-range mixes regimes where METRO's effect has
opposite signs. Always bucket by exact M.

Usage:
  python3 tools/analysis/per_m_breakdown.py <eplb_dir> <metro_dir> [label]
  python3 tools/analysis/per_m_breakdown.py --hash 8_8_1_64_32_2_dispatch_combine_0_0_{}_Qwen3-30B-A3B-8-128_g1 \
      --base results --base results/vllm_results_final
"""
import argparse
import collections
import re
import statistics
import sys
from pathlib import Path

CATS = ["attention", "gating", "routing", "dispatch", "expert", "combine"]
MOE = [c for c in CATS if c != "attention"]


def read(d):
    """-> {M: {cat: [us, ...]}}"""
    f = Path(d) / "server_breakdown.log"
    if not f.exists():
        return None
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(f):
        if not line.startswith("Breakdown"):
            continue
        kv = dict(re.findall(r"(\w+)=([-\d.]+)", line))
        if "M" not in kv:
            continue
        M = int(kv["M"])
        for c in CATS:
            if c + "_ns" in kv:
                acc[M][c].append(float(kv[c + "_ns"]) / 1000.0)
    return acc


def mean(acc, M, cats):
    return sum(statistics.mean(acc[M][c]) for c in cats if acc[M][c])


def report(eplb, metro, label=""):
    if label:
        print(f"\n=== {label} ===")
    Ms = sorted(set(eplb) & set(metro))
    if not Ms:
        print("  no overlapping M buckets")
        return
    w = "".join(f"{c:>9}" for c in CATS)
    print(f"{'M':>6} {'arm':6}{w}{'MoE':>9}{'gain%':>8}{'n':>9}")
    print("-" * (6 + 7 + 9 * len(CATS) + 9 + 8 + 9))
    for M in Ms:
        for tag, acc in (("EPLB", eplb), ("METRO", metro)):
            row = "".join(f"{statistics.mean(acc[M][c]) if acc[M][c] else 0:9.1f}" for c in CATS)
            print(f"{M:>6} {tag:6}{row}{mean(acc, M, MOE):9.1f}"
                  f"{'':>8}{len(acc[M]['expert']):9}")
        e, m = mean(eplb, M, MOE), mean(metro, M, MOE)
        g = (e - m) / e * 100 if e else float("nan")
        mark = "  <-- LOSS" if m > e else ""
        print(f"{'':>6} {'delta':6}"
              + "".join(f"{(statistics.mean(metro[M][c]) if metro[M][c] else 0) - (statistics.mean(eplb[M][c]) if eplb[M][c] else 0):+9.1f}" for c in CATS)
              + f"{m - e:+9.1f}{g:+8.1f}{'':>9}{mark}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--hash", help="dir-name template with {} where the threshold goes")
    ap.add_argument("--base", action="append", default=[],
                    help="repeatable; each base is reported separately")
    a = ap.parse_args()

    if a.hash:
        for base in (a.base or ["results"]):
            e, m = read(Path(base) / a.hash.format(0)), read(Path(base) / a.hash.format(256))
            if e is None or m is None:
                print(f"\n=== {base} ===\n  missing server_breakdown.log "
                      f"(eplb={'ok' if e else 'MISSING'}, metro={'ok' if m else 'MISSING'})")
                continue
            report(e, m, base)
    elif len(a.dirs) >= 2:
        e, m = read(a.dirs[0]), read(a.dirs[1])
        if e is None or m is None:
            sys.exit("missing server_breakdown.log in one of the dirs")
        report(e, m, a.dirs[2] if len(a.dirs) > 2 else "")
    else:
        ap.error("give two dirs, or --hash with --base")


if __name__ == "__main__":
    main()
