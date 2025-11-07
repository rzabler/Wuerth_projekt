# orsy_eda.py
# ==========================================
# ORSY-fokussierte EDA für den Wuerth-Datensatz
# - Zielvariable: orsy_buyer = 1, wenn mind. eines der ORSY-Flags aktiv ist
# - Plots & Aggregationen werden erstellt
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Config ------------------
DATA_PATH = "data/dataset_wuerth.csv"   # Pfad zur CSV anpassen, falls nötig
PLOT_DIR  = "eda_plots"            # Hierhin werden PNGs gespeichert
DATE_COLS = ["last_buy", "cust_since"]

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)


# -------------- Helpers -------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def header(txt: str):
    line = "=" * len(txt)
    print(f"\n{line}\n{txt}\n{line}")

def save_plot(name: str):
    ensure_dir(PLOT_DIR)
    path = os.path.join(PLOT_DIR, name)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"-> Plot gespeichert: {path}")


# -------------- Load data -----------------
header("Daten laden")
df = pd.read_csv(DATA_PATH)
print("Form:", df.shape)
print("Spalten:", list(df.columns))

# -------------- Parse dates ---------------
header("Datumsfelder konvertieren")
for c in DATE_COLS:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
print(df.dtypes[df.columns.isin(DATE_COLS)])

# -------------- Zielvariable --------------
header("Zielvariable ORSY (aus Flags)")
for c in ["flag_new_orsyshelf"]:
    if c in df.columns:
        df[c] = (df[c].fillna(0) > 0).astype(int)

df["orsy_buyer"] = (
    df.get("flag_new_orsyshelf", 0)
    
) > 0
df["orsy_buyer"] = df["orsy_buyer"].astype(int)

print(df["orsy_buyer"].value_counts(dropna=False))
print(f"Anteil ORSY-Käufer gesamt: {df['orsy_buyer'].mean():.2%}")

# -------------- Feature-Extras ------------
header("Feature Engineering")
today = pd.Timestamp.today().normalize()
if "last_buy" in df.columns:
    df["days_since_last_buy"] = (today - df["last_buy"]).dt.days
if "cust_since" in df.columns:
    df["customer_age_years"] = ((today - df["cust_since"]).dt.days / 365.25).round(2)

flag_cols = [c for c in df.columns if c.startswith("flag_")]
if flag_cols:
    df["digital_flags_count"] = df[flag_cols].fillna(0).sum(axis=1)
    df["digital_flags_share"] = df["digital_flags_count"] / len(flag_cols)

print("Neu angelegte Spalten (falls vorhanden):",
      [c for c in ["days_since_last_buy","customer_age_years","digital_flags_count","digital_flags_share"] if c in df.columns])

# -------------- Basics & NA ---------------
header("Fehlende Werte (Top)")
na = df.isna().sum()
na = na[na > 0].sort_values(ascending=False)
print(na.head(15))

header("Statistische Übersicht (nur numerisch)")
print(df.select_dtypes(include=["number"]).describe().T)

# -------------- Aggregationen -------------
header("Gruppenauswertungen")

if {"market_seg","sales","orsy_buyer"}.issubset(df.columns):
    seg = df.groupby("market_seg", as_index=False).agg(
        customers=("cust_id","count"),
        orsy_rate=("orsy_buyer","mean"),
        sales_sum=("sales","sum"),
        sales_mean=("sales","mean")
    ).sort_values(["orsy_rate","sales_sum"], ascending=False)
    print("\nUmsatz/ORSY nach Marktsegment (Top 10):")
    print(seg.head(10))
    seg.to_csv("agg_market_seg.csv", index=False)
    print("-> Export: agg_market_seg.csv")

if {"region","sales","orsy_buyer"}.issubset(df.columns):
    reg = df.groupby("region", as_index=False).agg(
        customers=("cust_id","count"),
        orsy_rate=("orsy_buyer","mean"),
        sales_sum=("sales","sum"),
        sales_mean=("sales","mean")
    ).sort_values("orsy_rate", ascending=False)
    print("\nUmsatz/ORSY nach Region:")
    print(reg)
    reg.to_csv("agg_region.csv", index=False)
    print("-> Export: agg_region.csv")

# -------------- Plots ---------------------
header("Plots (ORSY-Fokus)")
sns.set_theme()

# 0) ORSY-Anteil gesamt
plt.figure(figsize=(4,4))
plt.bar(["kein ORSY","ORSY"], [1 - df["orsy_buyer"].mean(), df["orsy_buyer"].mean()])
plt.title("Anteil ORSY-Käufer gesamt")
plt.ylabel("Anteil")
save_plot("00_orsy_rate_total.png")

# 1) ORSY-Anteil pro Region
if {"region","orsy_buyer"}.issubset(df.columns):
    region_rate = df.groupby("region")["orsy_buyer"].mean().sort_index()
    plt.figure(figsize=(8,5))
    sns.barplot(x=region_rate.index.astype(str), y=region_rate.values)
    plt.title("Anteil ORSY-Käufer pro Region")
    plt.xlabel("Region (interne Codes 11–18)")
    plt.ylabel("Anteil ORSY-Kauf")
    save_plot("01_orsy_rate_by_region.png")

# 2) ORSY-Anteil pro Marktsegment
if {"market_seg","orsy_buyer"}.issubset(df.columns):
    seg_rate = df.groupby("market_seg")["orsy_buyer"].mean().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=seg_rate.index.astype(str), y=seg_rate.values)
    plt.title("Anteil ORSY-Käufer pro Marktsegment")
    plt.xlabel("Marktsegment")
    plt.ylabel("Anteil ORSY-Kauf")
    plt.xticks(rotation=45)
    save_plot("02_orsy_rate_by_segment.png")

# 3) Top Branch Offices nach ORSY-Anteil
if {"branch_office","orsy_buyer"}.issubset(df.columns):
    bo_rate = (
        df.dropna(subset=["branch_office"])
          .groupby("branch_office")["orsy_buyer"].mean()
          .sort_values(ascending=False)
          .head(20)
    )
    plt.figure(figsize=(10,6))
    sns.barplot(y=bo_rate.index, x=bo_rate.values)
    plt.title("Top 20 Branch Offices nach ORSY-Anteil")
    plt.xlabel("Anteil ORSY-Kauf")
    plt.ylabel("Branch Office")
    save_plot("03_orsy_rate_by_branch_top20.png")

# 4) Umsatzvergleich: ORSY vs. kein ORSY
if {"sales","orsy_buyer"}.issubset(df.columns):
    plt.figure(figsize=(6,5))
    sns.boxplot(x="orsy_buyer", y="sales", data=df)
    plt.title("Umsatz: ORSY-Käufer (1) vs. Nicht-Käufer (0)")
    plt.xlabel("ORSY-Kauf")
    plt.ylabel("sales")
    save_plot("04_sales_distribution_by_orsy.png")

# 5) Digitalisierungsgrad vs. ORSY
if {"digital_flags_share","orsy_buyer"}.issubset(df.columns):
    plt.figure(figsize=(6,5))
    sns.boxplot(x="orsy_buyer", y="digital_flags_share", data=df)
    plt.title("Digitalisierungs-Score: ORSY vs. nicht ORSY")
    plt.xlabel("ORSY-Kauf")
    plt.ylabel("digital_flags_share")
    save_plot("05_digitalization_by_orsy.png")

# 6) Aktivität: E-Shop Logins / Calls / Orders
for var in ["eshop_logins_count", "calls_count", "orders_count"]:
    if {var,"orsy_buyer"}.issubset(df.columns):
        plt.figure(figsize=(6,5))
        sns.boxplot(x="orsy_buyer", y=var, data=df)
        plt.title(f"{var}: ORSY-Käufer vs. Nicht-Käufer")
        plt.xlabel("ORSY-Kauf")
        save_plot(f"06_{var}_by_orsy.png")

# 7) ORSY nach Unternehmensgröße (Binning)
if {"emp_count","orsy_buyer"}.issubset(df.columns):
    bins = [-1, 1, 5, 20, 50, 200, 1000]
    labels = ["0-1", "2-5", "6-20", "21-50", "51-200", "200+"]
    df["emp_size_bin"] = pd.cut(df["emp_count"], bins=bins, labels=labels)
    size_rate = df.groupby("emp_size_bin")["orsy_buyer"].mean()
    plt.figure(figsize=(8,5))
    sns.barplot(x=size_rate.index.astype(str), y=size_rate.values)
    plt.title("Anteil ORSY-Käufer nach Unternehmensgröße")
    plt.xlabel("Mitarbeiter (Binned)")
    plt.ylabel("Anteil ORSY-Kauf")
    save_plot("07_orsy_rate_by_emp_size.png")

# 8) Korrelationen mit ORSY (nur numerisch)
num_df = df.select_dtypes(include=[np.number]).copy()
if "orsy_buyer" in num_df.columns and num_df.shape[1] > 1:
    corr_with_orsy = num_df.corr()["orsy_buyer"].sort_values(ascending=False)
    plt.figure(figsize=(8,10))
    sns.heatmap(corr_with_orsy.to_frame(), annot=True, cmap="coolwarm", center=0)
    plt.title("Korrelationen mit ORSY-Kauf (orsy_buyer)")
    save_plot("08_corr_with_orsy.png")

header("Fertig ✅ – Plots in eda_plots/, Aggregationen als CSV")