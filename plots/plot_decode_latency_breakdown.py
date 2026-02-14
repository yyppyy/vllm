#!/usr/bin/env python3
"""
Analyze nsys profile data to generate decoding latency breakdown.

This script:
1. Exports nsys-rep files to SQLite format (or uses pre-exported files)
2. Identifies decode batches using marker events
3. Categorizes CUDA kernels into: attention, topk, communication, expert, others
4. Computes average latency excluding outliers
5. Generates visualization comparing TP vs EP

IMPORTANT - For macOS users:
Since nsys command-line tools are not available on macOS, you need to pre-export
the .nsys-rep files to SQLite format. Do this on the GPU machine or using Nsight Systems UI:

Method 1 - Using nsys command line (on GPU machine):
  cd results/vllm_results_dev/8_8_0_0_4_0_allgather_reducescatter_0_1/
  nsys export --type sqlite profile.nsys-rep
  # This creates profile.sqlite

  cd ../8_8_1_0_4_0_allgather_reducescatter_0_1/
  nsys export --type sqlite profile.nsys-rep
  # This creates profile.sqlite

Method 2 - Using Nsight Systems UI:
  1. Open profile.nsys-rep in Nsight Systems UI
  2. File -> Export -> SQLite
  3. Save as profile.sqlite in the same directory
  4. Repeat for both profile files

Once the .sqlite files exist, this script will use them automatically.
"""
import sqlite3
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re

from utils import set_paper_style, get_palette, HATCHES

# Profile file paths
TP_PROFILE = Path("../results/vllm_results_dev/8_8_0_0_4_0_allgather_reducescatter_0_1/profile.nsys-rep")
EP_PROFILE = Path("../results/vllm_results_dev/8_8_1_0_4_0_allgather_reducescatter_0_1/profile.nsys-rep")
OUTPUT_FILE = Path(__file__).parent / "decode_latency_breakdown.pdf"

# Layer markers (start and end of a single transformer layer)
LAYER_START_MARKER = "ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn"
LAYER_END_MARKER = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_1"

# Analyze only the last N layers to ensure we're in decode phase (not prefill)
ANALYZE_LAST_N_LAYERS = 100

# Skip first N layers from those as outliers (warmup)
SKIP_FIRST_N_LAYERS = 5


def export_nsys_to_sqlite(nsys_rep_path):
    """
    Export nsys-rep file to SQLite database.

    If nsys command is not available, the SQLite file should be pre-exported.
    See instructions in the script header for how to export using nsys-ui or command line.
    """
    sqlite_path = nsys_rep_path.with_suffix('.sqlite')

    if sqlite_path.exists():
        print(f"Using existing SQLite database: {sqlite_path}")
        return sqlite_path

    # Try to export using nsys command if available
    print(f"SQLite file not found: {sqlite_path}")
    print(f"Attempting to export {nsys_rep_path.name} to SQLite...")

    cmd = [
        'nsys', 'export',
        '--type', 'sqlite',
        '--output', str(sqlite_path),
        str(nsys_rep_path)
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Export complete: {sqlite_path}")
        return sqlite_path
    except FileNotFoundError:
        print("\nERROR: nsys command not found!")
        print("\nTo use this script on macOS, you need to pre-export the .nsys-rep files to SQLite format.")
        print("Please do ONE of the following:\n")
        print("Option 1: Export using nsys command line (on GPU machine):")
        print(f"  nsys export --type sqlite --output {sqlite_path} {nsys_rep_path}\n")
        print("Option 2: Export using Nsight Systems UI:")
        print(f"  1. Open {nsys_rep_path.name} in Nsight Systems UI")
        print("   2. File -> Export -> SQLite")
        print(f"  3. Save as {sqlite_path.name}")
        print(f"  4. Copy the .sqlite file to: {sqlite_path.parent}/\n")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Error exporting nsys profile: {e}")
        print(f"stderr: {e.stderr}")
        raise


def categorize_kernel(kernel_name):
    """
    Categorize CUDA kernel based on its name.

    Categories:
    - attention: Attention-related kernels (flash attention, gemm for QK/AV)
    - topk: Top-K selection kernels
    - communication: Inter-GPU communication (NCCL, all-reduce, etc.)
    - expert: Expert computation (MoE-related)
    - others: Everything else
    """
    name_lower = kernel_name.lower()

    # Communication patterns
    if any(pattern in name_lower for pattern in [
        'nccl', 'allreduce', 'allgather', 'reducescatter',
        'alltoall', 'p2p', 'send', 'recv', 'broadcast'
    ]):
        return 'communication'

    # Attention patterns
    if any(pattern in name_lower for pattern in [
        'flash', 'fmha', 'attention', 'attn',
        'scaled_dot_product', 'softmax'
    ]):
        return 'attention'

    # TopK patterns
    if any(pattern in name_lower for pattern in [
        'topk', 'top_k', 'select_top', 'argmax', 'argtop'
    ]):
        return 'topk'

    # Expert/MoE patterns
    if any(pattern in name_lower for pattern in [
        'moe', 'expert', 'gate', 'router', 'routing'
    ]):
        return 'expert'

    # Check for specific gemm patterns that might be expert FFN
    # Expert FFNs often use specific gemm configurations
    if 'gemm' in name_lower and any(pattern in name_lower for pattern in [
        'sliced', 'grouped', 'splitk'
    ]):
        return 'expert'

    return 'others'


def load_cuda_events(sqlite_path):
    """Load CUDA kernel events from SQLite database."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # First, explore the schema to find the right table and columns
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cursor.fetchall()]

    # Try to find kernel table
    kernel_table = None
    for table_name in tables:
        if 'KERNEL' in table_name.upper():
            kernel_table = table_name
            break

    if kernel_table is None:
        print(f"ERROR: No kernel table found. Available tables: {tables}")
        conn.close()
        raise ValueError("No kernel table found in SQLite database")

    print(f"Using table: {kernel_table}")

    # Get column names
    cursor.execute(f"PRAGMA table_info({kernel_table})")
    columns = {col[1]: col[2] for col in cursor.fetchall()}  # name: type
    print(f"Available columns: {list(columns.keys())[:20]}...")  # Show first 20

    # Find the right columns for name, start, end
    # Common variations:
    name_col = None
    for col in ['demangledName', 'shortName', 'name', 'kernelName']:
        if col in columns:
            name_col = col
            break

    start_col = None
    for col in ['start', 'startTime', 'timestamp', 'startNs']:
        if col in columns:
            start_col = col
            break

    end_col = None
    for col in ['end', 'endTime', 'endNs']:
        if col in columns:
            end_col = col
            break

    if not all([name_col, start_col, end_col]):
        print(f"ERROR: Could not find required columns")
        print(f"  Name column: {name_col}")
        print(f"  Start column: {start_col}")
        print(f"  End column: {end_col}")
        conn.close()
        raise ValueError("Missing required columns")

    print(f"Using columns: name={name_col}, start={start_col}, end={end_col}")

    # Check if we need to join with StringIds table for kernel names
    # In nsys exports, kernel names are often stored as IDs that reference StringIds table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='StringIds';")
    has_string_ids = cursor.fetchone() is not None

    if has_string_ids:
        print("Found StringIds table - will join to get actual kernel names")

        # Verify StringIds table structure
        cursor.execute("PRAGMA table_info(StringIds)")
        string_cols = {col[1]: col[2] for col in cursor.fetchall()}
        print(f"StringIds columns: {list(string_cols.keys())}")

        # Check sample data from StringIds
        cursor.execute("SELECT * FROM StringIds LIMIT 3")
        samples = cursor.fetchall()
        print(f"Sample StringIds entries: {samples}")

        # Join with StringIds to get actual kernel names
        query = f"""
        SELECT
            k.{start_col} as start_ns,
            k.{end_col} as end_ns,
            COALESCE(s.value, CAST(k.{name_col} AS TEXT)) as name
        FROM {kernel_table} k
        LEFT JOIN StringIds s ON k.{name_col} = s.id
        ORDER BY k.{start_col}
        """
    else:
        # Direct query without join
        query = f"""
        SELECT
            {start_col} as start_ns,
            {end_col} as end_ns,
            {name_col} as name
        FROM {kernel_table}
        ORDER BY {start_col}
        """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Ensure name column is string type
    df['name'] = df['name'].fillna('').astype(str)

    # Compute duration in nanoseconds (keep everything in ns for precision)
    df['duration_ns'] = df['end_ns'] - df['start_ns']

    print(f"Loaded {len(df)} CUDA kernel events")

    # Verify we have actual kernel names, not just IDs
    sample_names = df['name'].head(20).to_list()
    print(f"\nSample kernel names:")
    for i, name in enumerate(sample_names[:10]):
        print(f"  {i+1}. {name[:100]}...")  # Truncate long names

    # Check if we got actual names or just numeric IDs
    numeric_count = sum(1 for name in sample_names if name.isdigit())
    if numeric_count > len(sample_names) / 2:
        print(f"\nWARNING: {numeric_count}/{len(sample_names)} sample names are numeric IDs!")
        print("Kernel names may not have been properly resolved from StringIds table.")
        print("\nTrying alternative columns...")

        # Try shortName or mangledName if they exist
        for alt_col in ['shortName', 'mangledName']:
            if alt_col in columns and alt_col != name_col:
                print(f"Attempting to use {alt_col} instead...")
                # Recursive approach might be too complex, just warn for now
                print(f"Please manually update the script to use column: {alt_col}")
                break

    return df


def find_decode_layers(df):
    """
    Find decode layer boundaries using marker events.

    Returns list of tuples: (layer_idx, start_time_ns, end_time_ns)
    """
    print(f"\nSearching for layer markers...")
    print(f"  Start marker: {LAYER_START_MARKER}")
    print(f"  End marker: {LAYER_END_MARKER}")

    # Find layer start events (try exact match first, then partial)
    start_events = df[df['name'].str.contains(LAYER_START_MARKER, regex=False, na=False)]
    if len(start_events) == 0:
        # Try partial match with just the key part
        partial_marker = LAYER_START_MARKER.split('_')[0:3]  # e.g., "ampere_bf16_s16816gemm"
        partial_pattern = '_'.join(partial_marker)
        print(f"  No exact match for start marker, trying partial: {partial_pattern}")
        start_events = df[df['name'].str.contains(partial_pattern, regex=False, na=False)]

    # Find layer end events
    end_events = df[df['name'].str.contains(LAYER_END_MARKER, regex=False, na=False)]
    if len(end_events) == 0:
        # Try partial match
        partial_marker = LAYER_END_MARKER.split('_')[0:3]  # e.g., "triton_red_fused"
        partial_pattern = '_'.join(partial_marker)
        print(f"  No exact match for end marker, trying partial: {partial_pattern}")
        end_events = df[df['name'].str.contains(partial_pattern, regex=False, na=False)]

    print(f"Found {len(start_events)} layer start markers")
    print(f"Found {len(end_events)} layer end markers")

    if len(start_events) == 0 or len(end_events) == 0:
        print("\nWARNING: No layer markers found!")
        print("Layer start marker:", LAYER_START_MARKER)
        print("Layer end marker:", LAYER_END_MARKER)
        print("\nSearching for similar kernel names...")

        # Search for similar patterns
        print("\nKernels containing 'ampere' or 'gemm':")
        similar = df[df['name'].str.contains('ampere|gemm', case=False, regex=True, na=False)]
        if len(similar) > 0:
            print(similar['name'].unique()[:10])
        else:
            print("  None found")

        print("\nKernels containing 'triton' or 'fused':")
        similar = df[df['name'].str.contains('triton|fused', case=False, regex=True, na=False)]
        if len(similar) > 0:
            print(similar['name'].unique()[:10])
        else:
            print("  None found")

        return []

    all_layers = []
    start_times = start_events['start_ns'].values
    end_times = end_events['end_ns'].values

    # Match start and end events
    # For each start event, find the next end event
    for i, start_time in enumerate(start_times):
        # Find the first end event that comes after this start event
        matching_ends = end_times[end_times > start_time]
        if len(matching_ends) > 0:
            end_time = matching_ends[0]
            all_layers.append((i, start_time, end_time))

    print(f"Identified {len(all_layers)} total layers")

    # Take only the last N layers (decode phase)
    if len(all_layers) > ANALYZE_LAST_N_LAYERS:
        layers = all_layers[-ANALYZE_LAST_N_LAYERS:]
        print(f"Analyzing last {ANALYZE_LAST_N_LAYERS} layers (decode phase)")
    else:
        layers = all_layers
        print(f"Using all {len(layers)} layers (less than {ANALYZE_LAST_N_LAYERS})")

    # Print layer durations for inspection (convert to ms for display)
    if layers:
        print("\nSample layer durations (ms):")
        # Show first 5 and last 5 of the selected layers
        for i, start, end in layers[:5]:
            print(f"  Layer {i}: {(end - start)/1e6:.2f} ms")
        if len(layers) > 10:
            print("  ...")
            for i, start, end in layers[-5:]:
                print(f"  Layer {i}: {(end - start)/1e6:.2f} ms")

    return layers


def compute_latency_breakdown(df, layers, skip_first_n=SKIP_FIRST_N_LAYERS):
    """
    Compute latency breakdown for each layer, then average.

    Args:
        df: DataFrame with CUDA events (times in nanoseconds)
        layers: List of (layer_idx, start_time_ns, end_time_ns) tuples
        skip_first_n: Number of initial layers to skip as warmup outliers

    Returns: dict of category -> average latency in ms (converted for readability)
    """
    if len(layers) <= skip_first_n:
        print(f"WARNING: Only {len(layers)} layers found, but skipping {skip_first_n}")
        skip_first_n = max(0, len(layers) - 10)  # Keep at least 10 layers if possible

    # Skip outlier layers (warmup)
    layers_to_analyze = layers[skip_first_n:]
    print(f"\nAnalyzing {len(layers_to_analyze)} layers (skipped first {skip_first_n} as warmup)")

    # Debug: check time ranges
    print(f"\nDEBUG: Time range analysis")
    print(f"  DataFrame event time range:")
    print(f"    Min: {df['start_ns'].min()/1e6:.2f} ms ({df['start_ns'].min():.0f} ns)")
    print(f"    Max: {df['end_ns'].max()/1e6:.2f} ms ({df['end_ns'].max():.0f} ns)")
    print(f"  Layer time range:")
    if layers_to_analyze:
        first_layer = layers_to_analyze[0]
        last_layer = layers_to_analyze[-1]
        print(f"    First layer: {first_layer[1]/1e6:.2f} - {first_layer[2]/1e6:.2f} ms")
        print(f"    Last layer:  {last_layer[1]/1e6:.2f} - {last_layer[2]/1e6:.2f} ms")

        # Check overlap
        layer_min = min(l[1] for l in layers_to_analyze)
        layer_max = max(l[2] for l in layers_to_analyze)
        df_min = df['start_ns'].min()
        df_max = df['end_ns'].max()

        if layer_max < df_min or layer_min > df_max:
            print(f"\n  WARNING: No overlap between layer times and event times!")
            print(f"    Layer range: {layer_min/1e6:.2f} - {layer_max/1e6:.2f} ms")
            print(f"    Event range: {df_min/1e6:.2f} - {df_max/1e6:.2f} ms")

    # Collect latency breakdown for each layer
    layer_breakdowns = []

    # Debug: show details for first layer
    show_debug = True

    for idx, (layer_idx, start_time, end_time) in enumerate(layers_to_analyze):
        # Get all events that overlap with this layer (times in ns)
        # An event overlaps if: event_start < layer_end AND event_end > layer_start
        layer_events = df[(df['start_ns'] < end_time) & (df['end_ns'] > start_time)]

        if show_debug and idx == 0:
            print(f"\nDEBUG: First layer analysis")
            print(f"  Layer time range: {start_time/1e6:.2f} - {end_time/1e6:.2f} ms")
            print(f"  Duration: {(end_time - start_time)/1e6:.2f} ms")
            print(f"  Events in layer: {len(layer_events)}")
            print(f"  Sample events:")
            for i, (_, event) in enumerate(layer_events.head(20).iterrows()):
                cat = categorize_kernel(event['name'])
                print(f"    {i+1}. {event['name'][:80]:80s} | {event['duration_ns']/1e6:.3f}ms | {cat}")

            # Show category distribution
            cat_counts = defaultdict(int)
            for _, event in layer_events.iterrows():
                cat = categorize_kernel(event['name'])
                cat_counts[cat] += 1
            print(f"\n  Category distribution:")
            for cat, count in sorted(cat_counts.items()):
                print(f"    {cat}: {count} kernels")

        # Categorize and sum latencies (in nanoseconds, convert to ms at the end)
        breakdown = defaultdict(float)
        for _, event in layer_events.iterrows():
            category = categorize_kernel(event['name'])
            breakdown[category] += event['duration_ns']

        layer_breakdowns.append(breakdown)

    # Check if we got any data
    if not layer_breakdowns:
        print("\nERROR: No layer breakdown data collected!")
        return {'attention': 0.0, 'topk': 0.0, 'communication': 0.0, 'expert': 0.0, 'others': 0.0, 'total': 0.0}

    # Compute total latency for each layer
    total_latencies_ns = [sum(bd.values()) for bd in layer_breakdowns]
    total_latencies_ms = [t / 1e6 for t in total_latencies_ns]

    # Calculate statistics for outlier detection
    median_latency = np.median(total_latencies_ns)
    mean_latency = np.mean(total_latencies_ns)
    std_latency = np.std(total_latencies_ns)

    print(f"\nAll layer statistics (before outlier removal):")
    print(f"  Total layers: {len(total_latencies_ns)}")
    print(f"  Median: {median_latency / 1e6:.2f} ms")
    print(f"  Mean:   {mean_latency / 1e6:.2f} ms")
    print(f"  Std:    {std_latency / 1e6:.2f} ms")
    print(f"  Min:    {np.min(total_latencies_ns) / 1e6:.2f} ms")
    print(f"  Max:    {np.max(total_latencies_ns) / 1e6:.2f} ms")

    # Filter outliers: only keep layers with latency <= median
    # This removes slow outliers while keeping the stable, typical layers
    stable_indices = [i for i, lat in enumerate(total_latencies_ns) if lat <= median_latency]

    print(f"\nOutlier filtering:")
    print(f"  Keeping layers with latency <= median ({median_latency / 1e6:.2f} ms)")
    print(f"  Stable layers: {len(stable_indices)} / {len(total_latencies_ns)}")
    print(f"  Removed: {len(total_latencies_ns) - len(stable_indices)} outliers")

    if not stable_indices:
        print("\nWARNING: No stable layers found! Using all layers.")
        stable_indices = list(range(len(layer_breakdowns)))

    # Debug: show breakdown for first few stable layers
    print(f"\nBreakdown for first 3 stable layers:")
    for i, idx in enumerate(stable_indices[:3]):
        bd = layer_breakdowns[idx]
        total = sum(bd.values()) / 1e6
        print(f"  Layer {idx}: total={total:.2f}ms", end="")
        if total > 0:
            bd_ms = {k: v/1e6 for k, v in bd.items()}
            print(f" -> {bd_ms}")
        else:
            print(" (no events!)")

    # Average only across stable (non-outlier) layers
    categories = ['attention', 'topk', 'communication', 'expert', 'others']
    avg_breakdown = {}

    for cat in categories:
        values = [layer_breakdowns[i].get(cat, 0.0) for i in stable_indices]
        avg_breakdown[cat] = np.mean(values) / 1e6 if values else 0.0  # Convert to ms

    # Compute statistics on stable layers only
    stable_latencies_ns = [total_latencies_ns[i] for i in stable_indices]
    avg_breakdown['total'] = np.mean(stable_latencies_ns) / 1e6
    avg_breakdown['std'] = np.std(stable_latencies_ns) / 1e6
    avg_breakdown['min'] = np.min(stable_latencies_ns) / 1e6
    avg_breakdown['max'] = np.max(stable_latencies_ns) / 1e6

    # Print final statistics (stable layers only)
    print(f"\nFinal layer latency statistics (stable layers only):")
    print(f"  Mean: {avg_breakdown['total']:.2f} ms")
    print(f"  Std:  {avg_breakdown['std']:.2f} ms")
    print(f"  Min:  {avg_breakdown['min']:.2f} ms")
    print(f"  Max:  {avg_breakdown['max']:.2f} ms")

    return avg_breakdown


def analyze_profile(profile_path, label):
    """Analyze a single profile and return latency breakdown."""
    print(f"\n{'='*60}")
    print(f"Analyzing {label}: {profile_path.name}")
    print(f"{'='*60}")

    # Export to SQLite
    sqlite_path = export_nsys_to_sqlite(profile_path)

    # Load CUDA events
    df = load_cuda_events(sqlite_path)

    # Find decode layers
    layers = find_decode_layers(df)

    if not layers:
        print(f"ERROR: No layers found for {label}")
        return None

    # Compute breakdown
    breakdown = compute_latency_breakdown(df, layers)

    print(f"\nLatency breakdown for {label} (per decode layer):")
    print(f"  Attention:      {breakdown['attention']:.2f} ms ({breakdown['attention']/breakdown['total']*100:.1f}%)")
    print(f"  TopK:           {breakdown['topk']:.2f} ms ({breakdown['topk']/breakdown['total']*100:.1f}%)")
    print(f"  Communication:  {breakdown['communication']:.2f} ms ({breakdown['communication']/breakdown['total']*100:.1f}%)")
    print(f"  Expert:         {breakdown['expert']:.2f} ms ({breakdown['expert']/breakdown['total']*100:.1f}%)")
    print(f"  Others:         {breakdown['others']:.2f} ms ({breakdown['others']/breakdown['total']*100:.1f}%)")
    print(f"  Total:          {breakdown['total']:.2f} ms")

    return breakdown


def plot_comparison(tp_breakdown, ep_breakdown):
    """Create stacked bar chart comparing TP and EP latency breakdowns."""
    set_paper_style()

    categories = ['attention', 'topk', 'communication', 'expert', 'others']
    category_labels = {
        'attention': 'Attention',
        'topk': 'Top-K',
        'communication': 'Communication',
        'expert': 'Expert',
        'others': 'Others'
    }

    # Prepare data
    systems = ['Tensor Parallel', 'Expert Parallel']
    tp_values = [tp_breakdown.get(cat, 0.0) for cat in categories]
    ep_values = [ep_breakdown.get(cat, 0.0) for cat in categories]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Get colors
    colors = get_palette(len(categories), name="tableau10")

    # Width of bars
    width = 0.5
    x = np.arange(len(systems))

    # Create stacked bars
    bottom_tp = 0.0
    bottom_ep = 0.0

    for i, cat in enumerate(categories):
        tp_val = tp_values[i]
        ep_val = ep_values[i]

        # TP bar
        ax.bar(0, tp_val, width, bottom=bottom_tp,
               color=colors[i], edgecolor='black', linewidth=1,
               hatch=HATCHES[i], label=category_labels[cat])

        # EP bar
        ax.bar(1, ep_val, width, bottom=bottom_ep,
               color=colors[i], edgecolor='black', linewidth=1,
               hatch=HATCHES[i])

        bottom_tp += tp_val
        bottom_ep += ep_val

    # Formatting
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel('Latency per Layer (ms)')
    ax.set_title('Decode Layer Latency Breakdown: TP vs EP')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # Add total latency annotations on top of bars
    ax.text(0, bottom_tp, f'{bottom_tp:.1f}ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(1, bottom_ep, f'{bottom_ep:.1f}ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add speedup annotation
    speedup = bottom_tp / bottom_ep if bottom_ep > 0 else 0
    ax.text(0.5, max(bottom_tp, bottom_ep) * 1.05, f'{speedup:.2f}x',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, format='pdf', bbox_inches='tight')
    print(f"\nSaved plot to {OUTPUT_FILE}")


def main():
    # Analyze both profiles
    tp_breakdown = analyze_profile(TP_PROFILE, "Tensor Parallel")
    ep_breakdown = analyze_profile(EP_PROFILE, "Expert Parallel")

    if tp_breakdown is None or ep_breakdown is None:
        print("\nERROR: Failed to analyze one or both profiles")
        return

    # Create comparison plot
    plot_comparison(tp_breakdown, ep_breakdown)

    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON - Per Decode Layer Latency")
    print("="*70)
    print(f"{'Category':<15} {'TP (ms)':<12} {'EP (ms)':<12} {'Diff (ms)':<12} {'Speedup':<10}")
    print("-"*70)

    categories = ['attention', 'topk', 'communication', 'expert', 'others', 'total']
    for cat in categories:
        tp_val = tp_breakdown.get(cat, 0.0)
        ep_val = ep_breakdown.get(cat, 0.0)
        diff = tp_val - ep_val
        speedup = tp_val / ep_val if ep_val > 0 else float('inf')
        print(f"{cat.capitalize():<15} {tp_val:<12.2f} {ep_val:<12.2f} {diff:<+12.2f} {speedup:<10.2f}x")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    tp_total = tp_breakdown.get('total', 0.0)
    ep_total = ep_breakdown.get('total', 0.0)
    overall_speedup = tp_total / ep_total if ep_total > 0 else 0

    print(f"Overall speedup: {overall_speedup:.2f}x")
    print(f"  TP takes {tp_total:.2f} ms per layer")
    print(f"  EP takes {ep_total:.2f} ms per layer")
    print(f"  EP is {tp_total - ep_total:+.2f} ms faster per layer")

    # Identify largest differences
    print("\nLargest differences:")
    diffs = [(cat, tp_breakdown.get(cat, 0) - ep_breakdown.get(cat, 0))
             for cat in ['attention', 'topk', 'communication', 'expert', 'others']]
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    for cat, diff in diffs[:3]:
        pct = abs(diff) / tp_total * 100 if tp_total > 0 else 0
        print(f"  {cat.capitalize()}: {diff:+.2f} ms ({pct:.1f}% of TP total)")


if __name__ == "__main__":
    main()
