#Check a full local run against the recorded experiment outputs#
import pandas as pd

from .dataset import ROOT


def verify_results():
    tables = [
        ("baseline", "metrics.csv", ["seed", "model", "labels"]),
        ("regularization", "metrics.csv", ["seed", "model", "labels"]),
        ("regularization", "selection.csv",
         ["seed", "labels", "max_depth", "min_child_weight", "reg_lambda", "learning_rate"]),
    ]
    for experiment, filename, keys in tables:
        relative = f"{experiment}/{filename}"
        tables_to_compare = []
        for directory in ("results", "runs"):
            path = ROOT / directory / relative
            if not path.is_file():
                raise SystemExit(f"Missing {path}. Run 'python -m kumo_benchmark all' first.")
            table = pd.read_csv(path).drop(columns=["seconds"], errors="ignore")
            tables_to_compare.append(table.set_index(keys, verify_integrity=True).sort_index())
        pd.testing.assert_frame_equal(*tables_to_compare, check_exact=False, rtol=0, atol=1e-6)
        print(f"Matched {relative} ({len(tables_to_compare[0])} rows)")
