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
LAYER_START_MARKER = "triton_red_fused__to_copy_mean_pow_2"
LAYER_END_MARKER = "ncclDevKernel_Reduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)"

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

    IMPORTANT: Check order matters! MoE/expert must come before attention because
    some MoE kernels (e.g., topkGatingSoftmax) contain 'softmax' which would
    otherwise match the attention category.
    """
    name_lower = kernel_name.lower()

    # Communication patterns (check first - NCCL is unambiguous)
    if any(pattern in name_lower for pattern in [
        'nccl', 'allreduce', 'allgather', 'reducescatter',
        'alltoall', 'p2p', 'send', 'recv', 'broadcast'
    ]):
        return 'communication'

    # Expert/MoE patterns (check BEFORE attention - moe::topkGatingSoftmax contains 'softmax')
    if any(pattern in name_lower for pattern in [
        'fused_moe', 'moe_align_block', 'count_and_sort_expert',
        'moe::', 'expert', 'router', 'routing'
    ]):
        return 'expert'

    # Check for specific gemm patterns that are expert FFN
    if 'gemm' in name_lower and any(pattern in name_lower for pattern in [
        'sliced', 'grouped', 'splitk'
    ]):
        return 'expert'

    # TopK patterns
    if any(pattern in name_lower for pattern in [
        'topk', 'top_k', 'select_top', 'argmax', 'argtop'
    ]):
        return 'topk'

    # Attention patterns
    if any(pattern in name_lower for pattern in [
        'flash', 'fmha', 'attention', 'attn',
        'scaled_dot_product', 'reshape_and_cache'
    ]):
        return 'attention'

    # Attention projection GEMV kernels (cuBLAS matrix-vector multiply for QKV/output projections)
    if 'gemvx' in name_lower or 'gemv' in name_lower:
        return 'attention'

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

    # Find the right columns for name, start, end, device
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

    # Find device/GPU ID column
    # IMPORTANT: Only use actual GPU device ID columns, NOT contextId or streamId.
    # NCCL kernels and compute kernels run on different streams/contexts on the same GPU,
    # so grouping by stream/context would incorrectly separate markers from their kernels.
    device_col = None
    for col in ['deviceId', 'device', 'gpuId']:
        if col in columns:
            device_col = col
            break
    if device_col is None:
        print("WARNING: No GPU device ID column found. All events will be treated as same device.")
        print(f"  Available columns: {list(columns.keys())}")

    if not all([name_col, start_col, end_col]):
        print(f"ERROR: Could not find required columns")
        print(f"  Name column: {name_col}")
        print(f"  Start column: {start_col}")
        print(f"  End column: {end_col}")
        conn.close()
        raise ValueError("Missing required columns")

    print(f"Using columns: name={name_col}, start={start_col}, end={end_col}, device={device_col}")

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
        device_select = f"k.{device_col} as device_id," if device_col else "0 as device_id,"
        stream_select = "k.streamId as stream_id," if 'streamId' in columns else "0 as stream_id,"
        pid_select = "k.globalPid as global_pid," if 'globalPid' in columns else "0 as global_pid,"
        query = f"""
        SELECT
            k.{start_col} as start_ns,
            k.{end_col} as end_ns,
            {device_select}
            {stream_select}
            {pid_select}
            COALESCE(s.value, CAST(k.{name_col} AS TEXT)) as name
        FROM {kernel_table} k
        LEFT JOIN StringIds s ON k.{name_col} = s.id
        ORDER BY k.{start_col}
        """
    else:
        # Direct query without join
        device_select = f"{device_col} as device_id," if device_col else "0 as device_id,"
        stream_select = "streamId as stream_id," if 'streamId' in columns else "0 as stream_id,"
        pid_select = "globalPid as global_pid," if 'globalPid' in columns else "0 as global_pid,"
        query = f"""
        SELECT
            {start_col} as start_ns,
            {end_col} as end_ns,
            {device_select}
            {stream_select}
            {pid_select}
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

    # Show device, stream, and globalPid distribution
    if 'device_id' in df.columns:
        device_counts = df['device_id'].value_counts().sort_index()
        print(f"\nEvents per device:")
        for device_id, count in device_counts.items():
            print(f"  Device {device_id}: {count} events")

    if 'global_pid' in df.columns:
        pid_counts = df['global_pid'].value_counts().sort_index()
        print(f"\nEvents per globalPid (each = one GPU rank):")
        for pid, count in pid_counts.items():
            print(f"  globalPid={pid}: {count} events")

        # In multi-GPU multi-process setups (torchrun), each rank sees its GPU as device 0.
        # All ranks' kernels appear with deviceId=0 but different globalPid values.
        # Filter to a single rank (the one with the most events) to get one GPU's view.
        if len(pid_counts) > 1:
            chosen_pid = pid_counts.idxmax()
            print(f"\nMulti-rank profile detected ({len(pid_counts)} ranks). "
                  f"Filtering to single rank: globalPid={chosen_pid} ({pid_counts[chosen_pid]} events)")
            df = df[df['global_pid'] == chosen_pid].reset_index(drop=True)
            print(f"After filtering: {len(df)} events")

    if 'stream_id' in df.columns:
        stream_counts = df['stream_id'].value_counts().sort_index()
        print(f"\nEvents per stream (after rank filtering):")
        for stream_id, count in stream_counts.items():
            print(f"  Stream {stream_id}: {count} events")

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
    Find decode layer boundaries using marker events, grouped by device and stream.

    Multiple CUDA streams may execute the same layer in parallel (one per batch element).
    We pick one representative stream per device to avoid counting the same work N times.

    Returns list of tuples: (device_id, stream_id, layer_idx, start_time_ns, end_time_ns)
    """
    print(f"\nSearching for layer markers...")
    print(f"  Start marker: {LAYER_START_MARKER}")
    print(f"  End marker: {LAYER_END_MARKER}")

    # Find layer start events
    start_events = df[df['name'].str.contains(LAYER_START_MARKER, regex=False, na=False)]
    print(f"  Found {len(start_events)} instances of start marker")

    # Find layer end events
    end_events = df[df['name'].str.contains(LAYER_END_MARKER, regex=False, na=False)]
    print(f"  Found {len(end_events)} instances of end marker")

    if len(start_events) == 0 or len(end_events) == 0:
        print("\nWARNING: No layer markers found!")
        print("Layer start marker:", LAYER_START_MARKER)
        print("Layer end marker:", LAYER_END_MARKER)
        return []

    # Show marker distribution by stream
    start_stream_counts = start_events['stream_id'].value_counts().sort_index()
    end_stream_counts = end_events['stream_id'].value_counts().sort_index()
    print(f"\nStart markers per stream:")
    for sid, cnt in start_stream_counts.items():
        print(f"  Stream {sid}: {cnt}")
    print(f"End markers per stream:")
    for sid, cnt in end_stream_counts.items():
        print(f"  Stream {sid}: {cnt}")

    # Group by device
    devices = sorted(df['device_id'].unique())
    print(f"\nDevices found: {devices}")

    all_layers = []

    # Process each device separately
    for device_id in devices:
        device_start = start_events[start_events['device_id'] == device_id]
        device_end = end_events[end_events['device_id'] == device_id]

        if len(device_start) == 0 or len(device_end) == 0:
            print(f"\nDevice {device_id}: Skipping - missing markers")
            continue

        # Find streams that have BOTH start and end markers
        start_streams = set(device_start['stream_id'].unique())
        end_streams = set(device_end['stream_id'].unique())
        common_streams = sorted(start_streams & end_streams)

        print(f"\nDevice {device_id}:")
        print(f"  Streams with start markers: {sorted(start_streams)}")
        print(f"  Streams with end markers: {sorted(end_streams)}")
        print(f"  Streams with both: {common_streams}")

        if not common_streams:
            # Start and end markers on different streams - try cross-stream matching
            print(f"  WARNING: No stream has both markers. Using cross-stream matching.")
            # Pick the stream with most start markers
            best_stream = device_start['stream_id'].value_counts().idxmax()
            stream_start = device_start[device_start['stream_id'] == best_stream]
            # Use end markers from any stream on this device
            stream_end = device_end
            chosen_stream = best_stream
        else:
            # Pick the stream with the most start markers
            best_stream = max(common_streams, key=lambda s: len(device_start[device_start['stream_id'] == s]))
            stream_start = device_start[device_start['stream_id'] == best_stream]
            stream_end = device_end[device_end['stream_id'] == best_stream]
            chosen_stream = best_stream

        print(f"  Chosen stream: {chosen_stream}")
        print(f"  Start markers on chosen stream: {len(stream_start)}")
        print(f"  End markers on chosen stream: {len(stream_end)}")

        # Match start and end events
        start_times = stream_start['start_ns'].values
        end_start_times = stream_end['start_ns'].values
        end_end_times = stream_end['end_ns'].values

        device_layers = []
        for i, start_time in enumerate(start_times):
            matching_end_indices = np.where(end_start_times > start_time)[0]
            if len(matching_end_indices) > 0:
                end_idx = matching_end_indices[0]
                end_time = end_end_times[end_idx]
                device_layers.append((device_id, chosen_stream, i, start_time, end_time))

        print(f"  Identified {len(device_layers)} layers")

        # Take only the last N layers for this device (decode phase)
        if len(device_layers) > ANALYZE_LAST_N_LAYERS:
            device_layers_to_use = device_layers[-ANALYZE_LAST_N_LAYERS:]
            print(f"  Using last {ANALYZE_LAST_N_LAYERS} layers (decode phase)")
        else:
            device_layers_to_use = device_layers
            print(f"  Using all {len(device_layers_to_use)} layers")

        # Show sample layer durations for this device
        if device_layers_to_use:
            sample_layers = device_layers_to_use[:3]
            print(f"  Sample layer durations:")
            for dev_id, stream_id, idx, start, end in sample_layers:
                print(f"    Layer {idx} (stream {stream_id}): {(end - start)/1e3:.1f} us")

        all_layers.extend(device_layers_to_use)

    print(f"\nTotal layers across all devices: {len(all_layers)}")

    return all_layers


def compute_latency_breakdown(df, layers, skip_first_n=SKIP_FIRST_N_LAYERS):
    """
    Compute latency breakdown for each layer, then average.

    Args:
        df: DataFrame with CUDA events (times in nanoseconds)
        layers: List of (device_id, stream_id, layer_idx, start_time_ns, end_time_ns) tuples
        skip_first_n: Number of initial layers to skip as warmup outliers PER DEVICE

    Returns: dict of category -> average latency in ms (converted for readability)
    """
    # Group layers by device and skip first N per device
    layers_by_device = defaultdict(list)
    for layer in layers:
        device_id = layer[0]
        layers_by_device[device_id].append(layer)

    # Skip first N layers per device
    layers_to_analyze = []
    for device_id, device_layers in layers_by_device.items():
        if len(device_layers) <= skip_first_n:
            print(f"WARNING: Device {device_id} has only {len(device_layers)} layers, but skipping {skip_first_n}")
            skip_n = max(0, len(device_layers) - 10)
        else:
            skip_n = skip_first_n

        layers_to_analyze.extend(device_layers[skip_n:])
        print(f"Device {device_id}: Using {len(device_layers[skip_n:])} layers (skipped first {skip_n} as warmup)")

    print(f"\nTotal layers to analyze: {len(layers_to_analyze)}")

    # Debug: check time ranges
    if layers_to_analyze:
        first_layer = layers_to_analyze[0]
        last_layer = layers_to_analyze[-1]
        # Tuple format: (device_id, stream_id, layer_idx, start_time_ns, end_time_ns)
        print(f"  First layer: dev={first_layer[0]} stream={first_layer[1]} idx={first_layer[2]} "
              f"{first_layer[3]/1e6:.2f} - {first_layer[4]/1e6:.2f} ms")
        print(f"  Last layer:  dev={last_layer[0]} stream={last_layer[1]} idx={last_layer[2]} "
              f"{last_layer[3]/1e6:.2f} - {last_layer[4]/1e6:.2f} ms")

    # Collect latency breakdown for each layer
    layer_breakdowns = []
    layer_signatures = []  # Track which categories are present in each layer

    # Track unique kernels per category across all layers
    category_kernels = defaultdict(set)

    # Debug: show details for first layer
    show_debug = True

    for idx, (device_id, stream_id, layer_idx, start_time, end_time) in enumerate(layers_to_analyze):
        # Get all events that overlap with this layer AND are on the same device+stream.
        # Filtering by stream ensures we only capture kernels from ONE batch element,
        # not from all concurrent streams (which would N-x duplicate the count).
        layer_events = df[
            (df['device_id'] == device_id) &
            (df['stream_id'] == stream_id) &
            (df['start_ns'] < end_time) &
            (df['end_ns'] > start_time)
        ]

        if show_debug and idx == 0:
            print(f"\nDEBUG: First layer analysis")
            print(f"  Device: {device_id}, Stream: {stream_id}")
            print(f"  Layer time range: {start_time/1e3:.1f} - {end_time/1e3:.1f} us")
            print(f"  Duration: {(end_time - start_time)/1e3:.1f} us")

            # Verify stream filtering is working
            all_overlap = df[
                (df['device_id'] == device_id) &
                (df['start_ns'] < end_time) &
                (df['end_ns'] > start_time)
            ]
            stream_dist = all_overlap['stream_id'].value_counts().sort_index()
            print(f"  All overlapping events (any stream): {len(all_overlap)}")
            print(f"  Per-stream breakdown of overlapping events:")
            for sid, cnt in stream_dist.items():
                print(f"    Stream {sid}: {cnt} events")
            print(f"  Events on chosen stream {stream_id}: {len(layer_events)}")
            # Check for overlapping events - print timestamps relative to layer start
            print(f"\n  ALL events in first layer (times relative to layer start):")
            for i, (_, event) in enumerate(layer_events.iterrows()):
                cat = categorize_kernel(event['name'])
                rel_start = (event['start_ns'] - start_time) / 1e3  # relative start in us
                rel_end = (event['end_ns'] - start_time) / 1e3  # relative end in us
                print(f"    {i+1:3d}. [{cat:15s}] {rel_start:8.1f} - {rel_end:8.1f}us ({event['duration_ns']/1e3:6.1f}us) "
                      f"stream={event['stream_id']} | {event['name'][:90]}")

            # Show category distribution (raw, before merging)
            cat_counts = defaultdict(int)
            cat_times = defaultdict(float)
            for _, event in layer_events.iterrows():
                cat = categorize_kernel(event['name'])
                cat_counts[cat] += 1
                cat_times[cat] += event['duration_ns']
            print(f"\n  Category distribution (raw, before merging):")
            for cat, count in sorted(cat_counts.items()):
                total_time_us = cat_times[cat] / 1e3
                print(f"    {cat:15s}: {count:3d} kernels, {total_time_us:7.1f} us total (sum of durations)")

        # Merge overlapping events of the same kernel name to get wall-clock durations.
        # CUDA Graph replay causes N concurrent instances of the same kernel to appear
        # with overlapping timestamps. We group by kernel name, then merge overlapping
        # intervals within each group. This handles both consecutive and interleaved patterns.
        events_by_name = defaultdict(list)
        for _, event in layer_events.iterrows():
            events_by_name[event['name']].append((event['start_ns'], event['end_ns']))

        merged_events = []  # list of (name, merged_start_ns, merged_end_ns)
        for name, intervals in events_by_name.items():
            # Sort intervals by start time and merge overlapping ones
            intervals.sort()
            merged_start, merged_end = intervals[0]
            for s, e in intervals[1:]:
                if s < merged_end:  # overlapping
                    merged_end = max(merged_end, e)
                else:
                    merged_events.append((name, merged_start, merged_end))
                    merged_start, merged_end = s, e
            merged_events.append((name, merged_start, merged_end))

        # Clip merged events to layer boundaries
        clipped_events = []
        for name, m_start, m_end in merged_events:
            c_start = max(m_start, start_time)
            c_end = min(m_end, end_time)
            if c_end > c_start:
                clipped_events.append((name, c_end - c_start))

        if show_debug and idx == 0:
            print(f"\n  After merging overlapping events: {len(merged_events)} merged (from {len(layer_events)} raw)")
            print(f"  After clipping to layer: {len(clipped_events)} events")
            merged_total = sum(d for _, d in clipped_events)
            print(f"  Merged total duration: {merged_total/1e3:.1f} us (vs wall-clock {(end_time - start_time)/1e3:.1f} us)")

        # Categorize and sum latencies from merged events
        breakdown = defaultdict(float)
        layer_kernel_categories = set()

        for name, duration_ns in clipped_events:
            category = categorize_kernel(name)
            breakdown[category] += duration_ns

            if duration_ns > 0:
                layer_kernel_categories.add(category)
                category_kernels[category].add(name)

        layer_breakdowns.append(breakdown)
        # Create signature: frozenset of categories with non-zero latency
        layer_signatures.append(frozenset(layer_kernel_categories))

    # Check if we got any data
    if not layer_breakdowns:
        print("\nERROR: No layer breakdown data collected!")
        return {'attention': 0.0, 'topk': 0.0, 'communication': 0.0, 'expert': 0.0, 'others': 0.0, 'total': 0.0}

    # Print unique kernels per category
    print(f"\n" + "="*70)
    print("KERNEL CATEGORIZATION - ALL UNIQUE KERNELS")
    print("="*70)
    categories = ['attention', 'topk', 'communication', 'expert', 'others']
    for cat in categories:
        kernels = sorted(category_kernels[cat])
        print(f"\n{cat.upper()} ({len(kernels)} unique kernels):")
        if len(kernels) == 0:
            print(f"  (no kernels)")
        else:
            for kernel in kernels:  # Show ALL kernels
                print(f"  - {kernel}")

    # Find the most common layer signature (kernel composition)
    from collections import Counter
    signature_counts = Counter(layer_signatures)
    most_common_signature, signature_count = signature_counts.most_common(1)[0]

    print(f"\n" + "="*70)
    print("LAYER SIGNATURE ANALYSIS")
    print("="*70)
    print(f"Total unique signatures: {len(signature_counts)}")
    print(f"Most common signature appears in: {signature_count} / {len(layer_signatures)} layers")
    print(f"Most common signature categories: {sorted(most_common_signature)}")

    # Show all signatures and their counts
    print(f"\nAll signatures:")
    for sig, count in signature_counts.most_common():
        print(f"  {sorted(sig)}: {count} layers")

    # Filter layers by signature: only keep layers with the most common signature
    signature_filtered_indices = [i for i, sig in enumerate(layer_signatures) if sig == most_common_signature]

    print(f"\nSignature filtering:")
    print(f"  Keeping layers with majority signature")
    print(f"  Filtered layers: {len(signature_filtered_indices)} / {len(layer_signatures)}")
    print(f"  Removed: {len(layer_signatures) - len(signature_filtered_indices)} layers with different composition")

    # Compute total latency for signature-filtered layers
    total_latencies_ns = [sum(layer_breakdowns[i].values()) for i in signature_filtered_indices]
    total_latencies_ms = [t / 1e6 for t in total_latencies_ns]

    # Calculate statistics for outlier detection
    median_latency = np.median(total_latencies_ns)
    mean_latency = np.mean(total_latencies_ns)
    std_latency = np.std(total_latencies_ns)

    print(f"\nAll layer statistics (after signature filter, before latency outlier removal):")
    print(f"  Total layers: {len(total_latencies_ns)}")
    print(f"  Median: {median_latency / 1e3:.1f} us ({median_latency / 1e6:.3f} ms)")
    print(f"  Mean:   {mean_latency / 1e3:.1f} us")
    print(f"  Std:    {std_latency / 1e3:.1f} us")
    print(f"  Min:    {np.min(total_latencies_ns) / 1e3:.1f} us")
    print(f"  Max:    {np.max(total_latencies_ns) / 1e3:.1f} us")

    # Filter latency outliers: only keep layers with latency <= median
    # This removes slow outliers while keeping the stable, typical layers
    # stable_local_indices: indices into total_latencies_ns array
    stable_local_indices = [i for i, lat in enumerate(total_latencies_ns) if lat <= median_latency]
    # stable_indices: indices into original layer_breakdowns array
    stable_indices = [signature_filtered_indices[i] for i in stable_local_indices]

    print(f"\nLatency outlier filtering:")
    print(f"  Keeping layers with latency <= median ({median_latency / 1e6:.2f} ms)")
    print(f"  Stable layers: {len(stable_indices)} / {len(total_latencies_ns)}")
    print(f"  Removed: {len(total_latencies_ns) - len(stable_indices)} outliers")

    if not stable_indices:
        print("\nWARNING: No stable layers found! Using all signature-filtered layers.")
        stable_indices = signature_filtered_indices
        stable_local_indices = list(range(len(signature_filtered_indices)))

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
    stable_latencies_ns = [total_latencies_ns[i] for i in stable_local_indices]
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
