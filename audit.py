import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from fairlearn.metrics import (MetricFrame, selection_rate, true_positive_rate,
                               false_positive_rate, demographic_parity_difference,
                               equalized_odds_difference)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.postprocessing import ThresholdOptimizer

import data

SEED = 7
FOLDS = 5
np.random.seed(SEED)
os.makedirs("results", exist_ok=True)


def make_model(kind):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=0.5)
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                      random_state=SEED)


def group_report(y, pred, sens):
    mf = MetricFrame(metrics={"selection rate": selection_rate,
                              "true positive rate": true_positive_rate,
                              "false positive rate": false_positive_rate,
                              "accuracy": accuracy_score},
                     y_true=y, y_pred=pred, sensitive_features=sens)
    return mf.by_group


def summarise(y, pred, sens):
    return {"accuracy": accuracy_score(y, pred),
            "dp_diff": demographic_parity_difference(y, pred, sensitive_features=sens),
            "eo_diff": equalized_odds_difference(y, pred, sensitive_features=sens)}


def run_cv(df, attr, drop_sensitive, mitigation, kind="gbm"):
    # cross validated predictions so the fairness numbers are not from one lucky split
    X = data.features(df, drop_sensitive)
    y = df["label"].values
    sens = df[attr].values
    pred = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        # scale with training fold statistics only
        scaler = StandardScaler().fit(X.iloc[tr])
        Xs = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
        if mitigation is None:
            m = make_model(kind).fit(Xs.iloc[tr], y[tr])
            pred[te] = m.predict(Xs.iloc[te])
        elif mitigation == "threshold":
            m = ThresholdOptimizer(estimator=make_model(kind), constraints="equalized_odds",
                                   prefit=False, predict_method="predict_proba")
            m.fit(Xs.iloc[tr], y[tr], sensitive_features=sens[tr])
            pred[te] = m.predict(Xs.iloc[te], sensitive_features=sens[te], random_state=SEED)
        else:
            cons = DemographicParity() if mitigation == "dp" else EqualizedOdds()
            m = ExponentiatedGradient(make_model("logreg"), constraints=cons, eps=0.02)
            m.fit(Xs.iloc[tr], y[tr], sensitive_features=sens[tr])
            pred[te] = m.predict(Xs.iloc[te], random_state=SEED)
    return pred


def main():
    df = data.load()
    print("rows", len(df), "approval rate %.2f" % df["label"].mean())
    print(df.groupby("sex")["label"].agg(["mean", "size"]))
    print(df.groupby("age_group")["label"].agg(["mean", "size"]))

    all_results = {}
    for attr in ["sex", "age_group"]:
        print("\n===== protected attribute:", attr, "=====")
        y = df["label"].values
        sens = df[attr].values
        settings = [
            ("gbm, all features", False, None, "gbm"),
            ("gbm, sensitive columns dropped", True, None, "gbm"),
            ("logreg, all features", False, None, "logreg"),
            ("exp gradient, demographic parity", False, "dp", "logreg"),
            ("exp gradient, equalized odds", False, "eo", "logreg"),
            ("threshold optimizer, equalized odds", False, "threshold", "gbm"),
        ]
        rows = {}
        for name, drop, mit, kind in settings:
            pred = run_cv(df, attr, drop, mit, kind)
            s = summarise(y, pred, sens)
            rows[name] = s
            print("\n%s\n  accuracy %.3f  demographic parity diff %.3f  equalized odds diff %.3f"
                  % (name, s["accuracy"], s["dp_diff"], s["eo_diff"]))
            print(group_report(y, pred, sens).round(3).to_string())
        all_results[attr] = rows

        fig, ax = plt.subplots(figsize=(8, 5))
        for name, s in rows.items():
            ax.scatter(s["eo_diff"], s["accuracy"], s=70)
            ax.annotate(name, (s["eo_diff"], s["accuracy"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("equalized odds difference (lower is fairer)")
        ax.set_ylabel("cross validated accuracy")
        ax.set_title("accuracy versus fairness by %s, German credit" % attr)
        plt.tight_layout()
        plt.savefig("results/tradeoff_%s.png" % attr, dpi=120)

    json.dump(all_results, open("results/metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
