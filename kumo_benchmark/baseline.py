import os
from .dataset import ARCHIVE, ROOT
from time import perf_counter
from zipfile import ZipFile

os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))
import numpy as np
import pandas as pd
import torch
import sdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from .plot_results import render


def main():
    torch.set_num_threads(4)
    with ZipFile(ARCHIVE) as archive:
        with archive.open(
            next(n for n in archive.namelist() if n.endswith(".xls"))
        ) as f:
            data = pd.read_excel(f, header=1)
    assert data.shape == (30000, 25) and data.ID.is_unique
    x = data.iloc[:, 1:-1].astype("float32")
    y = data.iloc[:, -1].astype(int)
    pool, test = train_test_split(
        data.index, test_size=6000, stratify=y, random_state=42
    )
    model = sdm.models.KumoTabular(task="classification", size="small", device="cpu")
    out = ROOT / "runs" / "baseline"
    out.mkdir(exist_ok=True, parents=True)
    rows = []
    for seed in (12, 36, 53):
        large, _ = train_test_split(
            pool, train_size=1000, stratify=y.loc[pool], random_state=seed
        )
        small, _ = train_test_split(
            large, train_size=200, stratify=y.loc[large], random_state=seed
        )
        assert set(small) <= set(large) and not set(large).intersection(test)
        for name, train in (
            ("Kumo Small", small),
            ("XGBoost", small),
            ("XGBoost", large),
        ):
            start = perf_counter()
            if name == "Kumo Small":
                gen = torch.Generator().manual_seed(seed)
                model.fit(
                    torch.tensor(x.loc[train].values),
                    torch.tensor(y.loc[train].values[:, None]),
                    num_estimators=1,
                    generator=gen,
                )
                batches = []
                for ids in np.array_split(test, 48):
                    probs = model.predict(torch.tensor(x.loc[ids].values))
                    batches.extend(probs.to_pandas()["1"].to_numpy())
                    if len(batches) == len(ids):
                        print(
                            "First batch", len(ids), perf_counter() - start, flush=True
                        )
                scores = np.array(batches)
            else:
                baseline = XGBClassifier(n_jobs=4, random_state=seed)
                baseline.fit(x.loc[train], y.loc[train])
                scores = baseline.predict_proba(x.loc[test])[:, 1]
            elapsed = perf_counter() - start
            rows.append(
                dict(
                    model=name,
                    labels=len(train),
                    seed=seed,
                    seconds=elapsed,
                    roc_auc=roc_auc_score(y.loc[test], scores),
                    average_precision=average_precision_score(y.loc[test], scores),
                )
            )
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    render(pd.DataFrame(rows), out)


if __name__ == "__main__":
    main()
