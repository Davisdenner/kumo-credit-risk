#Command-line entry point for preparing data and reproducing results
import argparse
import os

from .dataset import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["data", "baseline", "regularization", "all", "figures", "verify"])
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
    if args.command == "verify":
        from .verify import verify_results
        verify_results()
        return
    if args.command == "figures":
        from .plot_results import redraw
        redraw()
        return
    print("Verified dataset:", prepare(), flush=True)
    if args.command in ("baseline", "all"):
        from .baseline import main as baseline
        baseline()
    if args.command in ("regularization", "all"):
        from .regularization import main as regularization
        regularization()


if __name__ == "__main__":
    main()

