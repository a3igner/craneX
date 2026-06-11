"""
CRANE-X Pipeline Orchestrator.

Usage:
    python3 run_pipeline.py --ingest          # Poll EODHD news by topic
    python3 run_pipeline.py --daemon          # Loop every 15 min
"""

import sys
import os
import subprocess
import time
import signal
import argparse
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_ingest():
    """Run EODHD news ingestion."""
    print("=" * 60)
    print(f"[CRANE-X] EODHD Ingestion — {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)

    script = os.path.join(SCRIPT_DIR, 'eodhd_ingest.py')
    result = subprocess.run(
        [sys.executable or 'python3', script],
        capture_output=False,
        cwd=SCRIPT_DIR,
    )
    return result.returncode


def daemon_loop():
    """Run pipeline in a loop forever."""
    print("[CRANE-X] Starting daemon mode (every 15 minutes)")
    signal.signal(signal.SIGTERM, lambda sig, f: sys.exit(0))

    while True:
        run_ingest()
        print("[CRANE-X] Sleeping 15 minutes...")
        time.sleep(15 * 60)


def main():
    parser = argparse.ArgumentParser(description='CRANE-X Pipeline')
    parser.add_argument('--ingest', action='store_true', help='Run EODHD news ingestion')
    parser.add_argument('--daemon', action='store_true', help='Run continuously (15min loop)')

    args = parser.parse_args()

    if args.daemon:
        daemon_loop()
    elif args.ingest:
        sys.exit(run_ingest())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
