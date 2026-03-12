"""
Credit Risk Prediction Model
JPMorgan Chase - Risk Management & Credit Analytics Division
Binary Classification: Loan Default Probability Prediction
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle

# ─────────────────────────────────────────────
# 1. DATA GENERATION (Simulated Financial Data)
# ─────────────────────────────────────────────
np.random.seed(42)
n_samples = 5000

# Simulate 15+ financial attributes
data = {
    'annual_income':        np.random.lognormal(11, 0.5, n_samples),
    'loan_amount':          np.random.lognormal(9.5, 0.6, n_samples),
    'credit_score':         np.random.normal(680, 80, n_samples).clip(300, 850),
    'debt_to_income_ratio': np.random.beta(2, 5, n_samples) * 60,
    'employment_years':     np.random.exponential(5, n_samples).clip(0, 35),
    'num_open_accounts':    np.random.poisson(6, n_samples),
    'num_credit_inquiries': np.random.poisson(2, n_samples),
    'delinquency_2yrs':     np.random.poisson(0.5, n_samples),
    'loan_to_value_ratio':  np.random.beta(3, 2, n_samples) * 100,
    'monthly_debt_payments':np.random.lognormal(7, 0.5, n_samples),
    'savings_balance':      np.random.lognormal(9, 1.2, n_samples),
    'num_late_payments':    np.random.poisson(1, n_samples),
    'revolving_utilization':np.random.beta(2, 3, n_samples) * 100,
    'total_credit_limit':   np.random.lognormal(10.5, 0.8, n_samples),
    'previous_defaults':    np.random.binomial(3, 0.1, n_samples),
    'loan_term_months':     np.random.choice([12, 24, 36, 48, 60], n_samples),
    'interest_rate':        np.random.normal(8, 3, n_samples).clip(1.5, 25),
}

df = pd.DataFrame(data)

# Feature Engineering
df['payment_to_income_ratio'] = df['monthly_debt_payments'] / (df['annual_income'] / 12)
df['savings_to_loan_ratio']   = df['savings_balance'] / df['loan_amount']
df['credit_risk_score']       = (
    (850 - df['credit_score']) * 0.4 +
    df['debt_to_income_ratio'] * 0.3 +
    df['revolving_utilization'] * 0.2 +
    df['num_late_payments'] * 5
)

# Target: default probability based on risk factors
default_prob = (
    0.30 * (df['credit_score'] < 620).astype(int) +
    0.25 * (df['debt_to_income_ratio'] > 40).astype(int) +
    0.20 * (df['previous_defaults'] > 0).astype(int) +
    0.15 * (df['revolving_utilization'] > 80).astype(int) +
    0.10 * (df['num_late_payments'] > 2).astype(int)
).clip(0, 0.9)
df['default'] = (np.random.random(n_samples) < default_prob).astype(int)

print(f"Dataset Shape    : {df.shape}")
print(f"Default Rate     : {df['default'].mean()*100:.1f}%")
print(f"Feature Count    : {df.shape[1]-1}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != 'default']
X = df[feature_cols]
y = df['default']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
# Logistic Regression
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('model',   LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

# Random Forest
rf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('model',   RandomForestClassifier(n_estimators=200, max_depth=12,
                                        class_weight='balanced', random_state=42, n_jobs=-1))
])

lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
lr_pred  = lr_pipeline.predict(X_test)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
rf_pred  = rf_pipeline.predict(X_test)
rf_proba = rf_pipeline.predict_proba(X_test)[:, 1]

lr_acc  = accuracy_score(y_test, lr_pred)
lr_auc  = roc_auc_score(y_test, lr_proba)
rf_acc  = accuracy_score(y_test, rf_pred)
rf_auc  = roc_auc_score(y_test, rf_proba)

print(f"\nLogistic Regression  → Accuracy: {lr_acc:.4f} | AUC-ROC: {lr_auc:.4f}")
print(f"Random Forest        → Accuracy: {rf_acc:.4f} | AUC-ROC: {rf_auc:.4f}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_cv  = cross_val_score(rf_pipeline, X, y, cv=cv, scoring='roc_auc')
print(f"Random Forest CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
BLUE  = '#003087'
RED   = '#CC0000'
GOLD  = '#DAA520'
GRAY  = '#F5F5F5'

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor('white')
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel 1: ROC Curves ---
ax1 = fig.add_subplot(gs[0, 0])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
ax1.plot(fpr_lr, tpr_lr, color=BLUE,  lw=2, label=f'Logistic Reg (AUC={lr_auc:.3f})')
ax1.plot(fpr_rf, tpr_rf, color=RED,   lw=2, label=f'Random Forest (AUC={rf_auc:.3f})')
ax1.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC Curves')
ax1.legend(fontsize=9); ax1.set_xlim([0,1]); ax1.set_ylim([0,1.02])

# --- Panel 2: Confusion Matrix - RF ---
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['No Default','Default'], yticklabels=['No Default','Default'])
ax2.set(xlabel='Predicted', ylabel='Actual', title='Confusion Matrix (Random Forest)')

# --- Panel 3: Feature Importance ---
ax3 = fig.add_subplot(gs[0, 2])
rf_model   = rf_pipeline.named_steps['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)
importances.plot(kind='barh', ax=ax3, color=BLUE, edgecolor='white')
ax3.set(title='Top Feature Importances (RF)', xlabel='Importance Score')
ax3.tick_params(labelsize=8)

# --- Panel 4: Score Distribution ---
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(rf_proba[y_test==0], bins=40, alpha=0.6, color=BLUE, label='No Default', density=True)
ax4.hist(rf_proba[y_test==1], bins=40, alpha=0.6, color=RED,  label='Default',    density=True)
ax4.axvline(0.5, color='black', linestyle='--', lw=1.5, label='Threshold=0.5')
ax4.set(xlabel='Predicted Probability', ylabel='Density', title='Default Probability Distribution')
ax4.legend(fontsize=9)

# --- Panel 5: Precision-Recall ---
ax5 = fig.add_subplot(gs[1, 1])
prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_proba)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_proba)
ax5.plot(rec_lr, prec_lr, color=BLUE, lw=2, label='Logistic Reg')
ax5.plot(rec_rf, prec_rf, color=RED,  lw=2, label='Random Forest')
ax5.set(xlabel='Recall', ylabel='Precision', title='Precision-Recall Curves')
ax5.legend(fontsize=9)

# --- Panel 6: Model Comparison Bar ---
ax6 = fig.add_subplot(gs[1, 2])
metrics  = ['Accuracy', 'AUC-ROC']
lr_vals  = [lr_acc, lr_auc]
rf_vals  = [rf_acc, rf_auc]
x = np.arange(len(metrics))
w = 0.35
b1 = ax6.bar(x - w/2, lr_vals, w, label='Logistic Regression', color=BLUE, alpha=0.85, edgecolor='white')
b2 = ax6.bar(x + w/2, rf_vals, w, label='Random Forest',       color=RED,  alpha=0.85, edgecolor='white')
for bar in list(b1) + list(b2):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax6.set(title='Model Performance Comparison', ylabel='Score', ylim=[0.7, 1.0])
ax6.set_xticks(x); ax6.set_xticklabels(metrics)
ax6.legend(fontsize=9)

# --- Panel 7: Cross-Validation ---
ax7 = fig.add_subplot(gs[2, 0])
folds = [f'Fold {i+1}' for i in range(5)]
ax7.bar(folds, rf_cv, color=BLUE, alpha=0.85, edgecolor='white')
ax7.axhline(rf_cv.mean(), color=RED, linestyle='--', lw=2, label=f'Mean={rf_cv.mean():.3f}')
ax7.fill_between(range(5), rf_cv.mean()-rf_cv.std(), rf_cv.mean()+rf_cv.std(),
                 alpha=0.15, color=RED)
ax7.set(title='5-Fold Cross-Validation AUC', ylabel='AUC-ROC Score', ylim=[0.7, 1.0])
ax7.legend(fontsize=9)

# --- Panel 8: Credit Score vs Default Rate ---
ax8 = fig.add_subplot(gs[2, 1])
bins  = pd.cut(df['credit_score'], bins=10)
rates = df.groupby(bins, observed=True)['default'].mean() * 100
rates.plot(kind='bar', ax=ax8, color=BLUE, alpha=0.85, edgecolor='white')
ax8.set(title='Default Rate by Credit Score Band', xlabel='Credit Score Range',
        ylabel='Default Rate (%)')
ax8.tick_params(axis='x', rotation=45, labelsize=7)

# --- Panel 9: Debt-to-Income vs Default ---
ax9 = fig.add_subplot(gs[2, 2])
bins2  = pd.cut(df['debt_to_income_ratio'], bins=8)
rates2 = df.groupby(bins2, observed=True)['default'].mean() * 100
rates2.plot(kind='bar', ax=ax9, color=RED, alpha=0.85, edgecolor='white')
ax9.set(title='Default Rate by Debt-to-Income Ratio', xlabel='DTI Range', ylabel='Default Rate (%)')
ax9.tick_params(axis='x', rotation=45, labelsize=7)

# --- Main Title ---
fig.suptitle('Credit Risk Prediction Model — JPMorgan Chase Analytics Dashboard',
             fontsize=16, fontweight='bold', color=BLUE, y=1.01)

plt.savefig('/home/claude/credit_risk_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("\n✅ Dashboard saved.")

# ─────────────────────────────────────────────
# 6. SAVE RESULTS DICT FOR DOC
# ─────────────────────────────────────────────
results = {
    'lr_acc': lr_acc, 'lr_auc': lr_auc,
    'rf_acc': rf_acc, 'rf_auc': rf_auc,
    'rf_cv_mean': rf_cv.mean(), 'rf_cv_std': rf_cv.std(),
    'n_samples': n_samples, 'n_features': len(feature_cols),
    'default_rate': df['default'].mean(),
    'top_features': importances.tail(5).index.tolist()[::-1],
    'report_rf': classification_report(y_test, rf_pred, target_names=['No Default','Default'])
}

with open('/home/claude/results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("Results Summary:")
print(f"  Samples           : {n_samples:,}")
print(f"  Features          : {len(feature_cols)}")
print(f"  RF Accuracy       : {rf_acc*100:.1f}%")
print(f"  RF AUC-ROC        : {rf_auc:.2f}")
print(f"  LR Accuracy       : {lr_acc*100:.1f}%")
print(f"  LR AUC-ROC        : {lr_auc:.2f}")
