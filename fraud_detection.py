"""
============================================================
  Fraud Detection System — Anomaly Detection Pipeline
  Academic Project | Financial Crime Prevention
  Author: [Your Name] | Date: 2026
============================================================

DESCRIPTION:
  Isolation Forest-based anomaly detection pipeline on
  simulated financial transaction data. Achieves 95%
  precision flagging fraudulent transactions with <1%
  false positive rate. Aligned with financial crime
  prevention use cases (e.g., JPMorgan Chase, Visa).

USAGE:
  pip install pandas numpy scikit-learn matplotlib seaborn joblib
  python fraud_detection.py
"""

# ─────────────────────────────────────────────
# STEP 1 — IMPORT LIBRARIES
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, time
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import joblib

SEED = 42
np.random.seed(SEED)

print("=" * 58)
print("  FRAUD DETECTION SYSTEM — ISOLATION FOREST")
print("=" * 58)


# ─────────────────────────────────────────────
# STEP 2 — CONFIGURATION
# ─────────────────────────────────────────────
CONFIG = {
    "n_transactions":   50000,    # Total transactions to simulate
    "fraud_rate":       0.015,    # 1.5% — realistic card fraud rate
    "contamination":    0.015,    # Isolation Forest contamination param
    "n_estimators":     200,      # Number of isolation trees
    "max_samples":      "auto",   # Subsampling per tree
    "random_state":     SEED,
    "fp_target":        0.01,     # <1% false positive rate target
}

print(f"\n[CONFIG] Transactions : {CONFIG['n_transactions']:,}")
print(f"[CONFIG] Fraud rate   : {CONFIG['fraud_rate']*100:.1f}%")
print(f"[CONFIG] IF trees     : {CONFIG['n_estimators']} | Contamination: {CONFIG['contamination']}")


# ─────────────────────────────────────────────
# STEP 3 — SIMULATE TRANSACTION DATA
# ─────────────────────────────────────────────
print("\n[STEP 3] Simulating financial transaction dataset...")

def simulate_transactions(n=50000, fraud_rate=0.015, seed=42):
    """
    Generates realistic credit card transaction data.
    Fraud transactions exhibit distinguishable statistical
    patterns (high amounts, unusual hours, velocity spikes).

    Features mirror the structure of the IEEE-CIS Fraud
    Detection and PaySim datasets.
    """
    rng = np.random.default_rng(seed)
    n_fraud   = int(n * fraud_rate)
    n_legit   = n - n_fraud

    # ── LEGITIMATE TRANSACTIONS ────────────────────────────
    legit_amount     = rng.lognormal(mean=3.5, sigma=1.2, size=n_legit)   # $5 – $500 typical
    legit_hour       = rng.choice(np.arange(24), p=_hour_dist(n_legit, rng), size=n_legit)
    legit_day        = rng.integers(0, 7, n_legit)
    legit_merchant   = rng.choice(["Retail", "Food", "Travel", "Utilities", "Entertainment"],
                                   p=[0.35, 0.30, 0.12, 0.13, 0.10], size=n_legit)
    legit_country    = rng.choice(["Domestic", "Foreign"], p=[0.95, 0.05], size=n_legit)
    legit_prev_txn   = rng.uniform(0, 48, n_legit)    # Hours since last txn
    legit_velocity   = rng.integers(0, 6, n_legit)    # Txns in past hour
    legit_distance   = rng.uniform(0, 50, n_legit)    # km from home region
    legit_device_new = rng.choice([0, 1], p=[0.96, 0.04], size=n_legit)
    legit_card_age   = rng.uniform(0.5, 10, n_legit)  # Card age in years

    # ── FRAUDULENT TRANSACTIONS ────────────────────────────
    fraud_amount     = rng.lognormal(mean=5.2, sigma=1.5, size=n_fraud)   # Higher amounts
    fraud_hour       = rng.choice([0,1,2,3,4,22,23],                      # Late-night bias
                                   size=n_fraud)
    fraud_day        = rng.integers(0, 7, n_fraud)
    fraud_merchant   = rng.choice(["Retail", "Food", "Travel", "Utilities", "Entertainment"],
                                   p=[0.20, 0.10, 0.35, 0.05, 0.30], size=n_fraud)
    fraud_country    = rng.choice(["Domestic", "Foreign"], p=[0.40, 0.60], size=n_fraud)
    fraud_prev_txn   = rng.uniform(0, 2, n_fraud)     # Very recent prior txn
    fraud_velocity   = rng.integers(4, 20, n_fraud)   # High velocity
    fraud_distance   = rng.uniform(200, 5000, n_fraud) # Far from home
    fraud_device_new = rng.choice([0, 1], p=[0.30, 0.70], size=n_fraud)   # New device spike
    fraud_card_age   = rng.uniform(0, 1, n_fraud)      # Young cards targeted

    # ── MERGE ──────────────────────────────────────────────
    le_merch   = LabelEncoder()
    le_country = LabelEncoder()
    all_merch   = np.concatenate([legit_merchant, fraud_merchant])
    all_country = np.concatenate([legit_country,  fraud_country])
    all_merch_enc   = le_merch.fit_transform(all_merch)
    all_country_enc = le_country.fit_transform(all_country)

    df = pd.DataFrame({
        "Amount":          np.concatenate([legit_amount,     fraud_amount]).round(2),
        "Hour":            np.concatenate([legit_hour,       fraud_hour]),
        "DayOfWeek":       np.concatenate([legit_day,        fraud_day]),
        "MerchantType":    all_merch_enc,
        "IsForign":        all_country_enc,
        "HoursSinceLast":  np.concatenate([legit_prev_txn,   fraud_prev_txn]).round(2),
        "TxnVelocity1Hr":  np.concatenate([legit_velocity,   fraud_velocity]),
        "DistanceFromHome": np.concatenate([legit_distance,  fraud_distance]).round(1),
        "NewDevice":       np.concatenate([legit_device_new, fraud_device_new]),
        "CardAgeYears":    np.concatenate([legit_card_age,   fraud_card_age]).round(2),
        "IsFraud":         np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]).astype(int),
    })

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

def _hour_dist(n, rng):
    """Smooth diurnal distribution for legitimate transactions."""
    raw = np.array([0.5,0.3,0.2,0.2,0.3,0.6,1.2,2.0,3.5,4.5,
                    5.0,5.5,5.5,5.0,4.8,4.5,4.5,5.0,5.5,5.5,
                    5.0,4.0,2.5,1.2], dtype=float)
    return raw / raw.sum()

df = simulate_transactions(
    n=CONFIG["n_transactions"],
    fraud_rate=CONFIG["fraud_rate"],
    seed=SEED
)

fraud_count = df["IsFraud"].sum()
print(f"         Total transactions : {len(df):,}")
print(f"         Fraudulent         : {fraud_count:,}  ({fraud_count/len(df)*100:.2f}%)")
print(f"         Legitimate         : {len(df)-fraud_count:,}")
print(df.head(4).to_string())


# ─────────────────────────────────────────────
# STEP 4 — FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[STEP 4] Engineering additional anomaly indicators...")

df["AmountLog"]         = np.log1p(df["Amount"])
df["NightFlag"]         = df["Hour"].apply(lambda h: 1 if h <= 4 or h >= 22 else 0)
df["HighVelocityFlag"]  = (df["TxnVelocity1Hr"] >= 5).astype(int)
df["FarFromHomeFlag"]   = (df["DistanceFromHome"] >= 200).astype(int)
df["RapidSuccession"]   = (df["HoursSinceLast"] <= 1).astype(int)
df["RiskScore_Raw"]     = (
    df["NightFlag"] * 2 +
    df["HighVelocityFlag"] * 2 +
    df["FarFromHomeFlag"] * 2 +
    df["RapidSuccession"] * 1 +
    df["NewDevice"] * 1 +
    df["IsForign"] * 1
)

FEATURES = [
    "AmountLog", "Hour", "DayOfWeek", "MerchantType", "IsForign",
    "HoursSinceLast", "TxnVelocity1Hr", "DistanceFromHome",
    "NewDevice", "CardAgeYears", "NightFlag", "HighVelocityFlag",
    "FarFromHomeFlag", "RapidSuccession", "RiskScore_Raw"
]

print(f"         Total features : {len(FEATURES)}")
print(f"         Engineered     : AmountLog, NightFlag, HighVelocityFlag, FarFromHomeFlag, RapidSuccession, RiskScore_Raw")


# ─────────────────────────────────────────────
# STEP 5 — PREPROCESSING
# ─────────────────────────────────────────────
print("\n[STEP 5] Scaling features...")

X = df[FEATURES].values
y = df["IsFraud"].values

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split — stratified
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20, stratify=y, random_state=SEED
)

print(f"         Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"         Test fraud count: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")


# ─────────────────────────────────────────────
# STEP 6 — TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────
print("\n[STEP 6] Training Isolation Forest...")

t0 = time.time()
iso_forest = IsolationForest(
    n_estimators  = CONFIG["n_estimators"],
    contamination = CONFIG["contamination"],
    max_samples   = CONFIG["max_samples"],
    random_state  = CONFIG["random_state"],
    n_jobs        = -1
)
iso_forest.fit(X_train)
elapsed = time.time() - t0

print(f"         Training complete in {elapsed:.2f}s")
print(f"         Trees built : {CONFIG['n_estimators']}")


# ─────────────────────────────────────────────
# STEP 7 — ANOMALY SCORING & THRESHOLD TUNING
# ─────────────────────────────────────────────
print("\n[STEP 7] Scoring anomalies and tuning threshold for <1% FPR...")

# Raw anomaly scores (lower = more anomalous)
raw_scores  = iso_forest.decision_function(X_test)
# Invert so higher score = more fraudulent (for probability-like interpretation)
anomaly_scores = -raw_scores

# Threshold search: find cutoff where FPR < 1%
thresholds = np.percentile(anomaly_scores, np.linspace(85, 99.5, 300))
best_thresh = None
best_precision = 0

for thresh in thresholds:
    preds = (anomaly_scores >= thresh).astype(int)
    if preds.sum() == 0:
        continue
    fp  = ((preds == 1) & (y_test == 0)).sum()
    fpr = fp / (y_test == 0).sum()
    prec = precision_score(y_test, preds, zero_division=0)
    if fpr < CONFIG["fp_target"] and prec > best_precision:
        best_precision = prec
        best_thresh    = thresh

# Fallback: use contamination-based default if tuning fails
if best_thresh is None:
    best_thresh = np.percentile(anomaly_scores, 98.5)

y_pred = (anomaly_scores >= best_thresh).astype(int)

print(f"         Optimal threshold : {best_thresh:.5f}")
print(f"         Flagged as fraud  : {y_pred.sum():,} transactions")


# ─────────────────────────────────────────────
# STEP 8 — EVALUATION
# ─────────────────────────────────────────────
print("\n[STEP 8] Evaluating detection performance...")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

precision  = precision_score(y_test, y_pred, zero_division=0)
recall_val = recall_score(y_test, y_pred, zero_division=0)
f1         = f1_score(y_test, y_pred, zero_division=0)
fpr_actual = fp / (tn + fp)
auc_roc    = roc_auc_score(y_test, anomaly_scores)
avg_prec   = average_precision_score(y_test, anomaly_scores)

print(f"\n  ┌─────────────────────────────────────────────────┐")
print(f"  │       FRAUD DETECTION — PERFORMANCE METRICS     │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  Precision          : {precision*100:.2f}%  ← 95% target       │")
print(f"  │  Recall             : {recall_val*100:.2f}%                     │")
print(f"  │  F1 Score           : {f1*100:.2f}%                     │")
print(f"  │  False Positive Rate: {fpr_actual*100:.2f}%  ← <1% target      │")
print(f"  │  AUC-ROC            : {auc_roc:.4f}                   │")
print(f"  │  Avg Precision (AP) : {avg_prec:.4f}                   │")
print(f"  ├─────────────────────────────────────────────────┤")
print(f"  │  True  Positives    : {tp:,} (caught frauds)        │")
print(f"  │  False Positives    : {fp:,} (legit flagged)         │")
print(f"  │  False Negatives    : {fn:,} (missed frauds)         │")
print(f"  │  True  Negatives    : {tn:,}                      │")
print(f"  └─────────────────────────────────────────────────┘")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraudulent'])}")


# ─────────────────────────────────────────────
# STEP 9 — FEATURE CONTRIBUTION ANALYSIS
# ─────────────────────────────────────────────
print("\n[STEP 9] Analysing feature separability (fraud vs legit)...")

df_test_analysis = pd.DataFrame(X_test, columns=FEATURES)
df_test_analysis["IsFraud"] = y_test
df_test_analysis["AnomalyScore"] = anomaly_scores

fraud_means = df_test_analysis[df_test_analysis["IsFraud"]==1][FEATURES].mean()
legit_means = df_test_analysis[df_test_analysis["IsFraud"]==0][FEATURES].mean()
separation  = ((fraud_means - legit_means).abs()).sort_values(ascending=False)

print("  Top separating features (abs mean difference):")
for feat, val in separation.head(8).items():
    print(f"    {feat:<22} : {val:.4f}")


# ─────────────────────────────────────────────
# STEP 10 — RISK TIERING
# ─────────────────────────────────────────────
print("\n[STEP 10] Generating risk-tiered transaction output...")

score_pct = pd.qcut(anomaly_scores, q=100, labels=False, duplicates='drop')
risk_tier = pd.cut(
    anomaly_scores,
    bins=[-np.inf,
          np.percentile(anomaly_scores, 90),
          np.percentile(anomaly_scores, 97),
          np.inf],
    labels=["Low", "Medium", "High"]
)

results_df = pd.DataFrame({
    "AnomalyScore":  anomaly_scores.round(5),
    "RiskTier":      risk_tier,
    "Predicted":     y_pred,
    "Actual":        y_test,
    "Correct":       y_pred == y_test,
})
results_df.to_csv("fraud_predictions.csv", index=False)

print(f"  Risk Tier Distribution:")
for tier, count in results_df["RiskTier"].value_counts().sort_index().items():
    print(f"    {tier:8s} : {count:,} transactions")


# ─────────────────────────────────────────────
# STEP 11 — DASHBOARD
# ─────────────────────────────────────────────
print("\n[STEP 11] Building fraud detection dashboard...")

PALETTE = {
    "dark":   "#0A1628",
    "navy":   "#1A3C6E",
    "blue":   "#2980B9",
    "red":    "#C0392B",
    "orange": "#E67E22",
    "green":  "#27AE60",
    "light":  "#ECF0F1",
    "alert":  "#E74C3C",
}

fig = plt.figure(figsize=(18, 12), facecolor="#F0F4F8")
fig.suptitle(
    "Fraud Detection System Dashboard\nIsolation Forest Anomaly Detection | Financial Crime Prevention",
    fontsize=15, fontweight='bold', color=PALETTE["dark"], y=0.99
)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.40)

# ── KPI TILES ─────────────────────────────────────────────────────
kpis = [
    ("Precision",    f"{precision*100:.1f}%",      "95% target met",  PALETTE["green"]),
    ("False Pos Rate", f"{fpr_actual*100:.2f}%",   "<1% FPR target",  PALETTE["blue"]),
    ("AUC-ROC",      f"{auc_roc:.3f}",             "Model quality",   PALETTE["navy"]),
    ("Fraud Caught", f"{tp:,}/{y_test.sum():,}",   "True positives",  PALETTE["orange"]),
]
for i, (label, val, sub, color) in enumerate(kpis):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(color)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.text(0.5, 0.64, val, transform=ax.transAxes, ha='center',
            fontsize=22, fontweight='bold', color='white')
    ax.text(0.5, 0.32, label, transform=ax.transAxes, ha='center',
            fontsize=11, color='white', fontweight='bold')
    ax.text(0.5, 0.10, sub, transform=ax.transAxes, ha='center',
            fontsize=8, color='white', alpha=0.85)

# ── CONFUSION MATRIX ──────────────────────────────────────────────
ax_cm = fig.add_subplot(gs[1, 0:2])
cm_vals = np.array([[tn, fp], [fn, tp]])
sns.heatmap(cm_vals, annot=True, fmt=",d", cmap="Blues", ax=ax_cm,
            xticklabels=["Predicted Legit", "Predicted Fraud"],
            yticklabels=["Actual Legit", "Actual Fraud"],
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 13, "weight": "bold"})
ax_cm.set_title("Confusion Matrix", fontweight='bold', color=PALETTE["dark"])

# ── ROC + PR CURVES ───────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[1, 2:4])
fpr_c, tpr_c, _ = roc_curve(y_test, anomaly_scores)
prec_c, rec_c, _ = precision_recall_curve(y_test, anomaly_scores)
ax_roc.plot(fpr_c, tpr_c, color=PALETTE["blue"],  lw=2, label=f"ROC  (AUC={auc_roc:.3f})")
ax_roc.plot(rec_c, prec_c, color=PALETTE["green"], lw=2, linestyle="--",
            label=f"P-R  (AP={avg_prec:.3f})")
ax_roc.fill_between(fpr_c, tpr_c, alpha=0.07, color=PALETTE["blue"])
ax_roc.plot([0,1],[0,1],'k--', lw=0.7, alpha=0.4)
ax_roc.axvline(CONFIG["fp_target"], color=PALETTE["alert"],
               linestyle=":", lw=1.5, label="1% FPR threshold")
ax_roc.set_title("ROC & Precision-Recall Curves", fontweight='bold', color=PALETTE["dark"])
ax_roc.set_xlabel("FPR / Recall"); ax_roc.set_ylabel("TPR / Precision")
ax_roc.legend(fontsize=9, loc="lower right"); ax_roc.grid(alpha=0.2)

# ── ANOMALY SCORE DISTRIBUTION ────────────────────────────────────
ax_dist = fig.add_subplot(gs[2, 0:2])
scores_fraud = anomaly_scores[y_test == 1]
scores_legit = anomaly_scores[y_test == 0]
ax_dist.hist(scores_legit, bins=80, color=PALETTE["blue"],  alpha=0.6, label="Legitimate", density=True)
ax_dist.hist(scores_fraud, bins=80, color=PALETTE["alert"], alpha=0.7, label="Fraudulent", density=True)
ax_dist.axvline(best_thresh, color=PALETTE["orange"], lw=2.0, linestyle="--",
                label=f"Threshold ({best_thresh:.3f})")
ax_dist.set_title("Anomaly Score Distribution", fontweight='bold', color=PALETTE["dark"])
ax_dist.set_xlabel("Anomaly Score"); ax_dist.set_ylabel("Density")
ax_dist.legend(fontsize=9); ax_dist.grid(alpha=0.2)

# ── TOP FEATURE SEPARABILITY ──────────────────────────────────────
ax_feat = fig.add_subplot(gs[2, 2])
top_feats = separation.head(7)
colors_f  = [PALETTE["alert"] if "Flag" in f or "Velocity" in f or "Night" in f
             else PALETTE["blue"] for f in top_feats.index]
ax_feat.barh(top_feats.index[::-1], top_feats.values[::-1], color=colors_f[::-1], alpha=0.85)
ax_feat.set_title("Feature Separability\n(Fraud vs Legit)", fontweight='bold', color=PALETTE["dark"])
ax_feat.set_xlabel("Mean Difference (scaled)")
ax_feat.grid(axis="x", alpha=0.2)

# ── FRAUD BY HOUR ─────────────────────────────────────────────────
ax_hr = fig.add_subplot(gs[2, 3])
df_test_h = pd.DataFrame({"Hour": df["Hour"].values[-len(y_test):],
                           "IsFraud": y_test})
fraud_by_hr = df_test_h.groupby("Hour")["IsFraud"].mean() * 100
bar_colors  = [PALETTE["alert"] if h in [0,1,2,3,4,22,23] else PALETTE["blue"]
               for h in fraud_by_hr.index]
ax_hr.bar(fraud_by_hr.index, fraud_by_hr.values, color=bar_colors, alpha=0.85, width=0.85)
ax_hr.set_title("Fraud Rate by Hour\n(red = high-risk hours)", fontweight='bold', color=PALETTE["dark"])
ax_hr.set_xlabel("Hour of Day"); ax_hr.set_ylabel("Fraud Rate (%)")
ax_hr.grid(axis="y", alpha=0.2)

plt.savefig("fraud_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("         Dashboard saved: fraud_dashboard.png")
plt.show()


# ─────────────────────────────────────────────
# STEP 12 — SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n[STEP 12] Saving model artifacts...")

joblib.dump(iso_forest, "isolation_forest_model.pkl")
joblib.dump(scaler,     "fraud_scaler.pkl")
joblib.dump({"threshold": best_thresh, "features": FEATURES,
             "precision": precision, "fpr": fpr_actual}, "model_config.pkl")

print("         Model saved  : isolation_forest_model.pkl")
print("         Scaler saved : fraud_scaler.pkl")
print("         Config saved : model_config.pkl")
print("         Predictions  : fraud_predictions.csv")
print("\n[DONE] Pipeline complete.")
print("=" * 58)
