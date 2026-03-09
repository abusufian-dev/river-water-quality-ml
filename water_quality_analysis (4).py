# ============================================================
# 🌊 River Water Quality ML Analysis
# Author  : Abu Sufian (github.com/abusufian-dev)
# Dataset : NH4 monitoring data — Ukrainian rivers (1996-2019)
# Goal    : Predict ammonium (NH4) pollution at Station 32
#           using upstream station readings + time features
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. LOAD & CLEAN DATA ────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading and Cleaning Data")
print("=" * 50)

df = pd.read_csv('PB_1996_2019_NH4.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
df = df.dropna()
df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

print(f"Total records  : {len(df)}")
print(f"Date range     : {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Stations       : {sorted(df['ID_Station'].unique())}")
print(f"NH4 range      : {df['NH4'].min():.3f} – {df['NH4'].max():.3f} mg/dm³")
print(f"Safe limit (Ukraine): 0.5 mg/dm³\n")

# ── 2. EXPLORATORY DATA ANALYSIS ────────────────────────────
print("=" * 50)
print("STEP 2: Exploratory Data Analysis")
print("=" * 50)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Plot 1 — NH4 over time (all stations)
axes[0].plot(df['Date'], df['NH4'], alpha=0.4, color='steelblue', linewidth=0.8)
axes[0].axhline(y=0.5, color='red', linestyle='--', label='Safe limit (0.5 mg/dm³)')
axes[0].set_title('NH4 Levels Across All Stations Over Time (1996–2019)', fontsize=13)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('NH4 (mg/dm³)')
axes[0].legend()

# Plot 2 — NH4 by station (boxplot)
sns.boxplot(x='ID_Station', y='NH4', data=df, ax=axes[1], color='steelblue')
axes[1].axhline(y=0.5, color='red', linestyle='--', label='Safe limit')
axes[1].set_title('NH4 Distribution by Station', fontsize=13)
axes[1].set_xlabel('Station ID')
axes[1].set_ylabel('NH4 (mg/dm³)')
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()
print("EDA chart saved → eda_overview.png\n")

# ── 3. PREPARE DATA FOR MODELING ────────────────────────────
print("=" * 50)
print("STEP 3: Preparing Data for ML Model")
print("=" * 50)

# Group by Year+Month (stations don't measure on exact same day)
monthly = df.groupby(['Year', 'Month', 'ID_Station'])['NH4'].mean().reset_index()

# Pivot: rows = Year+Month, columns = station IDs
pivoted = monthly.pivot_table(
    index=['Year', 'Month'],
    columns='ID_Station',
    values='NH4'
).reset_index()
pivoted.columns.name = None

print(f"Pivoted shape  : {pivoted.shape}")
print(f"Stations found : {[c for c in pivoted.columns if isinstance(c, int)]}\n")

# ── 4. BUILD ML MODEL ────────────────────────────────────────
print("=" * 50)
print("STEP 4: Training ML Model — Predicting Station 32")
print("=" * 50)

# Station 32 is the most critically polluted — best target
# Use stations with best data coverage as features
FEATURE_STATIONS = [27, 28, 29, 23, 35]
TARGET_STATION   = 32

all_cols = ['Year', 'Month'] + FEATURE_STATIONS + [TARGET_STATION]
model_df = pivoted[all_cols].dropna()

print(f"Usable rows    : {len(model_df)}")
print(f"NH4 at Stn 32  : min={model_df[TARGET_STATION].min():.2f}, "
      f"max={model_df[TARGET_STATION].max():.2f}, "
      f"mean={model_df[TARGET_STATION].mean():.2f} mg/dm³\n")

X = model_df[['Year', 'Month'] + FEATURE_STATIONS]
y = model_df[TARGET_STATION]
X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Model 1 — Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
r2_rf   = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Model 2 — Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                max_depth=4, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
r2_gb   = r2_score(y_test, y_pred_gb)
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print(f"Random Forest      → R²: {r2_rf:.4f}  |  RMSE: {rmse_rf:.4f} mg/dm³")
print(f"Gradient Boosting  → R²: {r2_gb:.4f}  |  RMSE: {rmse_gb:.4f} mg/dm³\n")

# ── 5. FEATURE IMPORTANCE & CORRELATION ─────────────────────
print("=" * 50)
print("STEP 5: Feature Importance & Correlation Analysis")
print("=" * 50)

corr = X.corrwith(model_df[TARGET_STATION]).sort_values()
print("Correlation with Station 32 NH4:")
print(corr.to_string())
print()

# ── 6. RESEARCH FINDINGS VISUALIZATION ──────────────────────
print("=" * 50)
print("STEP 6: Generating Research Findings Chart")
print("=" * 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1 — Yearly trend at Station 32
yearly = model_df.groupby('Year')[TARGET_STATION].mean()
axes[0].plot(yearly.index, yearly.values,
             marker='o', color='steelblue', linewidth=2)
axes[0].axhline(y=0.5, color='red', linestyle='--', label='Safe limit (0.5)')
axes[0].axvline(x=2010, color='orange', linestyle='--', label='Crisis point (2010)')
axes[0].fill_between(yearly.index, yearly.values, 0.5,
    where=(yearly.values > 0.5), color='red', alpha=0.2, label='Danger zone')
axes[0].set_title('NH4 Pollution Crisis at Station 32', fontsize=13)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('NH4 mg/dm³')
axes[0].legend()

# Plot 2 — Correlation bar chart
colors = ['red' if v > 0 else 'steelblue' for v in corr.values]
axes[1].barh(corr.index, corr.values, color=colors)
axes[1].axvline(x=0, color='black', linewidth=0.8)
axes[1].set_title('What Correlates with Station 32 Pollution?', fontsize=13)
axes[1].set_xlabel('Correlation coefficient')

plt.suptitle('River Water Quality Research — Ammonium (NH4) Analysis',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('research_findings.png', dpi=150, bbox_inches='tight')
plt.show()
print("Research chart saved → research_findings.png\n")

# ── 7. SUMMARY ───────────────────────────────────────────────
print("=" * 50)
print("RESEARCH SUMMARY")
print("=" * 50)
print(f"• Station 32 NH4 exceeded safe limit (0.5 mg/dm³) from 2010 onwards")
print(f"• Peak pollution: {yearly.max():.2f} mg/dm³ ({int(yearly.idxmax())}) — {yearly.max()/0.5:.0f}x the safe limit!")
print(f"• Year correlation: {corr['Year']:.3f} — pollution is TIME-driven, not upstream-driven")
print(f"• Best ML model (Random Forest): R² = {r2_rf:.4f}, RMSE = {rmse_rf:.4f} mg/dm³")
print(f"• Conclusion: Station 32 has an INDEPENDENT local pollution source")
print("=" * 50)
