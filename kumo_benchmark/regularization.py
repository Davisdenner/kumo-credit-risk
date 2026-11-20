# Follow-up using a fresh same-period holdout and training-only model selection
import os
from .dataset import ARCHIVE, ROOT
from zipfile import ZipFile

os.environ.setdefault("HF_HOME", str(ROOT / ".hf"))

import numpy as np
import pandas as pd
import sdm
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from xgboost import XGBClassifier
from .plot_results import render_followup


def main():
    with ZipFile(ARCHIVE) as archive:
        with archive.open(
            next(name for name in archive.namelist() if name.endswith(".xls"))
        ) as source:
            data = pd.read_excel(source, header=1)
    assert data.ID.is_unique and data.shape == (30000, 25)
    x = data.iloc[:, 1:-1].astype("float32")
    y = data.iloc[:, -1].astype(int)
    profiles = pd.util.hash_pandas_object(x, index=False)
    pool, previous_test = train_test_split(
        data.index, test_size=6000, stratify=y, random_state=42
    )
    samples = {}
    for seed in (12, 36, 53):
        large, _ = train_test_split(
            pool, train_size=1000, stratify=y.loc[pool], random_state=seed
        )
        small, _ = train_test_split(
            large, train_size=200, stratify=y.loc[large], random_state=seed
        )
        samples[seed] = (small, large)

    previous_ids = np.unique(
        np.concatenate([previous_test] + [large for _, large in samples.values()])
    )
    eligible = data.index[~profiles.isin(profiles.loc[previous_ids])]
    test, _ = train_test_split(
        eligible, train_size=6000, stratify=y.loc[eligible], random_state=2026
    )
    assert not set(previous_ids).intersection(test)
    assert not set(profiles.loc[test]).intersection(profiles.loc[previous_ids])

    torch.set_num_threads(4)
    kumo = sdm.models.KumoTabular(task="classification", size="small", device="cpu")
    results = []
    candidates = []
    for seed, sizes in samples.items():
        for train in sizes:
            xt, yt = x.loc[train], y.loc[train]
            folds = list(
                StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(
                    xt, yt, groups=profiles.loc[train]
                )
            )
            for fit, valid in folds:
                assert not set(profiles.loc[train[fit]]).intersection(
                    profiles.loc[train[valid]]
                )
            base = XGBClassifier(n_jobs=4, random_state=seed)
            search = GridSearchCV(
                base,
                dict(
                    max_depth=[2, 3],
                    min_child_weight=[1, 5],
                    reg_lambda=[5, 20],
                    learning_rate=[0.05],
                ),
                cv=folds,
                scoring="roc_auc",
                n_jobs=1,
                error_score="raise",
            )
            search.fit(xt, yt)
            for i, params in enumerate(search.cv_results_["params"]):
                candidates.append(
                    dict(
                        seed=seed,
                        labels=len(train),
                        **params,
                        cv_auc=search.cv_results_["mean_test_score"][i],
                        selected=i == search.best_index_,
                    )
                )
            for name, model in [
                ("XGBoost default", base.fit(xt, yt)),
                ("XGBoost regularized", search.best_estimator_),
            ]:
                scores = model.predict_proba(x.loc[test])[:, 1]
                results.append(
                    dict(
                        seed=seed,
                        labels=len(train),
                        model=name,
                        train_auc=roc_auc_score(yt, model.predict_proba(xt)[:, 1]),
                        roc_auc=roc_auc_score(y.loc[test], scores),
                        average_precision=average_precision_score(y.loc[test], scores),
                    )
                )
            if len(train) == 200:
                kumo.fit(
                    torch.tensor(xt.values),
                    torch.tensor(yt.values[:, None]),
                    generator=torch.Generator().manual_seed(seed),
                    num_estimators=1,
                )
                scores = np.concatenate(
                    [
                        kumo.predict(torch.tensor(x.loc[ids].values))
                        .to_pandas()["1"]
                        .to_numpy()
                        for ids in np.array_split(test, 48)
                    ]
                )
                results.append(
                    dict(
                        seed=seed,
                        labels=200,
                        model="Kumo Small",
                        roc_auc=roc_auc_score(y.loc[test], scores),
                        average_precision=average_precision_score(y.loc[test], scores),
                    )
                )
    out = ROOT / "runs" / "regularization"
    out.mkdir(exist_ok=True, parents=True)
    metrics = pd.DataFrame(results)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(candidates).to_csv(out / "selection.csv", index=False)
    render_followup(metrics, out)
    print(metrics.groupby(["model", "labels"]).mean().round(4))


if __name__ == "__main__":
    main()
