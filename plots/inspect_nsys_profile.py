#!/usr/bin/env python3
"""
Helper script to inspect nsys profile data structure.

Use this to:
1. Export nsys-rep to SQLite
2. Explore available tables and columns
3. Find kernel name patterns
4. Verify batch marker events exist
"""
import sqlite3
import subprocess
import sys
from pathlib import Path
import pandas as pd


def export_nsys_to_sqlite(nsys_rep_path):
    """Export nsys-rep file to SQLite database."""
    sqlite_path = nsys_rep_path.with_suffix('.sqlite')

    if sqlite_path.exists():
        print(f"SQLite database already exists: {sqlite_path}")
        return sqlite_path

    print(f"Exporting {nsys_rep_path.name} to SQLite...")
    cmd = [
        'nsys', 'export',
        '--type', 'sqlite',
        '--output', str(sqlite_path),
        str(nsys_rep_path)
    ]

    subprocess.run(cmd, check=True)
    print(f"Export complete: {sqlite_path}")
    return sqlite_path


def inspect_database(sqlite_path):
    """Inspect SQLite database structure."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    print("\n" + "="*60)
    print("DATABASE TABLES")
    print("="*60)

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name, in tables:
        print(f"\nTable: {table_name}")

        # Get table schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        print(f"  Columns: {', '.join([col[1] for col in columns])}")

        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  Rows: {count}")

    conn.close()


def find_kernel_table(sqlite_path):
    """Find the table containing CUDA kernel events."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Common table names for CUDA kernels
    possible_tables = [
        'CUPTI_ACTIVITY_KIND_KERNEL',
        'CUDA_KERNEL_EXEC_API_TRACE',
        'CUDA_KERNEL_EVENTS',
        'CUPTI_KERNEL',
        'NVTX_EVENTS'
    ]

    print("\n" + "="*60)
    print("SEARCHING FOR KERNEL TABLE")
    print("="*60)

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    all_tables = [t[0] for t in cursor.fetchall()]

    # Try to find kernel-related tables
    kernel_tables = [t for t in all_tables if any(
        keyword in t.upper() for keyword in ['KERNEL', 'CUDA', 'CUPTI']
    )]

    print(f"Found {len(kernel_tables)} potential kernel tables:")
    for table in kernel_tables:
        print(f"  - {table}")

        # Try to query it
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [col[1] for col in cursor.fetchall()]
                print(f"    Columns: {columns[:10]}...")  # Show first 10 columns
        except Exception as e:
            print(f"    Error querying: {e}")

    conn.close()
    return kernel_tables


def search_kernel_names(sqlite_path, pattern, limit=20):
    """Search for kernel names matching a pattern."""
    conn = sqlite3.connect(sqlite_path)

    # Try different possible table/column combinations
    queries = [
        ("CUPTI_ACTIVITY_KIND_KERNEL", "demangledName"),
        ("CUPTI_ACTIVITY_KIND_KERNEL", "shortName"),
        ("CUPTI_ACTIVITY_KIND_KERNEL", "name"),
    ]

    for table, column in queries:
        try:
            query = f"""
            SELECT DISTINCT {column}
            FROM {table}
            WHERE {column} LIKE '%{pattern}%'
            LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)

            if not df.empty:
                print(f"\n" + "="*60)
                print(f"Found {len(df)} kernels matching '{pattern}' in {table}.{column}")
                print("="*60)
                for name in df[column]:
                    print(f"  {name}")
                conn.close()
                return True

        except Exception as e:
            continue

    print(f"\nNo kernels found matching '{pattern}'")
    conn.close()
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_nsys_profile.py <profile.nsys-rep|profile.sqlite> [search_pattern]")
        print("\nExample:")
        print("  python inspect_nsys_profile.py profile.nsys-rep")
        print("  python inspect_nsys_profile.py profile.sqlite gemm")
        sys.exit(1)

    profile_path = Path(sys.argv[1])
    search_pattern = sys.argv[2] if len(sys.argv) > 2 else None

    if not profile_path.exists():
        print(f"Error: File not found: {profile_path}")
        sys.exit(1)

    # Export to SQLite or use existing SQLite file
    if profile_path.suffix == '.sqlite':
        print(f"Using SQLite database: {profile_path}")
        sqlite_path = profile_path
    else:
        sqlite_path = export_nsys_to_sqlite(profile_path)

    # Inspect database structure
    inspect_database(sqlite_path)

    # Find kernel table
    find_kernel_table(sqlite_path)

    # Search for specific patterns if provided
    if search_pattern:
        search_kernel_names(sqlite_path, search_pattern)
    else:
        # Search for the batch markers
        print("\n" + "="*60)
        print("SEARCHING FOR BATCH MARKERS")
        print("="*60)
        print("\nBatch start marker:")
        search_kernel_names(sqlite_path, "ampere_bf16_s16816gemm", limit=5)
        print("\nBatch end marker:")
        search_kernel_names(sqlite_path, "triton_red_fused", limit=5)


if __name__ == "__main__":
    main()
