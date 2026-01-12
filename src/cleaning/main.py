import argparse
from src.cleaning.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="Modular Furniture Data Cleaner")
    
    parser.add_argument("--all", action="store_true", help="Run full pipeline (Prepare -> Dispatch -> Monitor -> Finalize)")
    parser.add_argument("--prepare", action="store_true", help="Only generate JSONL files")
    parser.add_argument("--dispatch", action="store_true", help="Upload and start batches (Respects queue limits)")
    parser.add_argument("--monitor", action="store_true", help="Check status of running batches")
    parser.add_argument("--finalize", action="store_true", help="Download results and merge to CSV")
    parser.add_argument("--smoke", type=int, help="Limit preparation to X rows for testing")
    
    args = parser.parse_args()
    orch = Orchestrator()

    if args.all:
        orch.run_all()
    else:
        if args.prepare:
            orch.run_preparation()
        if args.dispatch:
            orch.run_dispatch()
        if args.monitor:
            orch.run_monitoring()
        if args.finalize:
            orch.run_finalization()

if __name__ == "__main__":
    main()