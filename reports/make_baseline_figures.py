"""Reproduces the baseline pipeline from notebooks/03_model_training.ipynb (same split seeds, same
models) and saves the figures and tables used in the midterm report's Baseline Solution section.

Run from the repo root:  python reports/make_baseline_figures.py
Outputs go to reports/figures/ and reports/baseline_metrics.csv
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay, PrecisionRecallDisplay)

from src.preproc import prepare_data, create_preprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

FEATURES = ["Vehicle Type", "Pickup Location", "Drop Location", "Hour", "DayOfWeek", "Month", "Avg VTAT"]
NO_VTAT = FEATURES[:-1]

df = pd.read_csv(os.path.join(ROOT, "data", "uberData.csv"))


def split(X, y):
    # identical to the notebook: 70 / 15 / 15, stratified, seed 42
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def models(features):
    return {
        "Logistic regression": Pipeline([("preprocessor", create_preprocessor(features)),
                                         ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                                           random_state=42))]),
        "Random forest": Pipeline([("preprocessor", create_preprocessor(features)),
                                   ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                                         random_state=42, n_jobs=-1))]),
    }


def metrics(y, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, proba), "AP": average_precision_score(y, proba)}


rows = []
fitted = {}
for label, feats in [("with VTAT", FEATURES), ("no VTAT", NO_VTAT)]:
    X, y = prepare_data(df, features=feats)
    X_tr, X_va, X_te, y_tr, y_va, y_te = split(X, y)
    for name, pipe in models(feats).items():
        pipe.fit(X_tr, y_tr)
        fitted[(label, name)] = (pipe, X_va, X_te, y_va, y_te)
        rows.append({"Features": label, "Model": name, "Set": "validation", "Threshold": 0.5,
                     **metrics(y_va, pipe.predict_proba(X_va)[:, 1], 0.5)})
        print("fitted", label, name)

# Threshold sweep on validation for the selected model (random forest with VTAT), as in the notebook
rf, X_va, X_te, y_va, y_te = fitted[("with VTAT", "Random forest")]
rf_va = rf.predict_proba(X_va)[:, 1]
sweep = pd.DataFrame([{"Threshold": round(t, 2),
                       "Precision": precision_score(y_va, (rf_va >= t).astype(int), zero_division=0),
                       "Recall": recall_score(y_va, (rf_va >= t).astype(int), zero_division=0),
                       "F1": f1_score(y_va, (rf_va >= t).astype(int), zero_division=0)}
                      for t in np.arange(0.05, 0.96, 0.05)])
best_thr = float(sweep.loc[sweep["F1"].idxmax(), "Threshold"])
print("selected threshold:", best_thr)

# Final test evaluation at the selected threshold
rf_te = rf.predict_proba(X_te)[:, 1]
rows.append({"Features": "with VTAT", "Model": "Random forest", "Set": "test", "Threshold": best_thr,
             **metrics(y_te, rf_te, best_thr)})
lr, *_ = fitted[("with VTAT", "Logistic regression")]
lr_te = lr.predict_proba(X_te)[:, 1]
rows.append({"Features": "with VTAT", "Model": "Logistic regression", "Set": "test", "Threshold": 0.5,
             **metrics(y_te, lr_te, 0.5)})

table = pd.DataFrame(rows)
table.round(3).to_csv(os.path.join(ROOT, "reports", "baseline_metrics.csv"), index=False)
print(table.round(3).to_string(index=False))
cm = confusion_matrix(y_te, (rf_te >= best_thr).astype(int))
print("test confusion matrix (RF @ %.2f):\n%s" % (best_thr, cm))

# ---- Figure 1: validation PR curves + threshold sweep -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for (label, name), (pipe, Xv, _, yv, _) in fitted.items():
    style = "-" if label == "with VTAT" else ":"
    PrecisionRecallDisplay.from_predictions(yv, pipe.predict_proba(Xv)[:, 1], ax=axes[0],
                                            name=f"{name}, {label}", linestyle=style)
axes[0].axhline(y_va.mean(), color="gray", lw=0.8, ls="--")
axes[0].set_title("Validation precision-recall"); axes[0].legend(fontsize=7, loc="upper right")
axes[1].plot(sweep["Threshold"], sweep["Precision"], marker="o", ms=3, label="Precision")
axes[1].plot(sweep["Threshold"], sweep["Recall"], marker="o", ms=3, label="Recall")
axes[1].plot(sweep["Threshold"], sweep["F1"], marker="o", ms=3, label="F1")
axes[1].axvline(best_thr, color="gray", ls="--", lw=0.8, label="selected %.2f" % best_thr)
axes[1].set_xlabel("Decision threshold"); axes[1].set_title("Random forest: validation sweep"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_baseline_pr_threshold.png"), dpi=200); plt.close()

# ---- Figure 2: test confusion matrices, LR @0.5 and RF @ selected ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
ConfusionMatrixDisplay(confusion_matrix(y_te, (lr_te >= 0.5).astype(int)),
                       display_labels=["Completed", "Cancelled"]).plot(ax=axes[0], colorbar=False, values_format="d")
axes[0].set_title("Logistic regression (threshold 0.50)")
ConfusionMatrixDisplay(cm, display_labels=["Completed", "Cancelled"]).plot(ax=axes[1], colorbar=False, values_format="d")
axes[1].set_title("Random forest (threshold %.2f)" % best_thr)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_baseline_confusion_test.png"), dpi=200); plt.close()

# ---- Figure 3: what the models rely on ----------------------------------------------------------
names = rf.named_steps["preprocessor"].get_feature_names_out()
def origin(n):
    body = n.split("__", 1)[1]
    for c in ["Vehicle Type", "Pickup Location", "Drop Location"]:
        if body.startswith(c + "_"):
            return c
    return body
imp = pd.Series(rf.named_steps["classifier"].feature_importances_, index=names).groupby(origin).sum().sort_values()
coef = pd.Series(lr.named_steps["classifier"].coef_[0], index=names)
coef_num = coef[[n for n in names if n.startswith("numeric__")]].rename(lambda n: n.split("__")[1])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
axes[0].barh(imp.index, imp.values, color="#4C72B0"); axes[0].set_title("Random forest: importance")
axes[1].barh(coef_num.index, coef_num.values, color=["#C44E52" if v > 0 else "#4C72B0" for v in coef_num.values])
axes[1].axvline(0, color="k", lw=0.6); axes[1].set_title("Logistic regression: numeric coefficients")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig_baseline_importance.png"), dpi=200); plt.close()
print("importance:\n", imp.round(3).to_string()); print("numeric coefs:\n", coef_num.round(3).to_string())
print("done")
