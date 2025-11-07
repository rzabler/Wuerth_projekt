# plot_orsy_vs_sales_corrected.py
# ============================================================
# Plot: ORSY-Umsatz vs. tatsächlicher Gesamtumsatz
# Gesamtumsatz = Summe der Umsatzkomponenten
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------ Einstellungen ------------------
DATA_PATH = "dataset_wuerth.csv"   # Pfad ggf. anpassen
PLOT_DIR  = "eda_plots"

# ------------------ Daten laden --------------------
df = pd.read_csv(DATA_PATH)
print(f"Datensatz geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")

# ------------------ Umsatz berechnen ---------------
umsatz_components = [
    "rev_salesrep",
    "rev_branch_office",
    "rev_ebusiness",
    "rev_internal_staff",
    "rev_others"
]

# Sicherstellen, dass alle Spalten vorhanden sind
for col in umsatz_components + ["sales_orsy_relevant"]:
    if col not in df.columns:
        raise ValueError(f"Spalte '{col}' fehlt im Datensatz!")

df["total_revenue"] = df[umsatz_components].sum(axis=1)

# ------------------ ORSY-Anteil berechnen ----------
df["orsy_share"] = df["sales_orsy_relevant"] / df["total_revenue"]
df["orsy_share"] = df["orsy_share"].replace([float("inf"), -float("inf")], 0).fillna(0)

# ------------------ Statistische Übersicht ----------
print("\n=== Grundlegende Kennzahlen ===")
print(f"Ø Gesamtumsatz (Summe): {df['total_revenue'].mean():.2f} €")
print(f"Ø ORSY-Umsatz:          {df['sales_orsy_relevant'].mean():.2f} €")
print(f"Ø ORSY-Anteil:           {df['orsy_share'].mean() * 100:.2f} %")

# ------------------ Plot 1: Scatterplot -------------
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df.sample(min(5000, len(df))), 
    x="total_revenue", 
    y="sales_orsy_relevant", 
    alpha=0.5
)
sns.regplot(
    data=df.sample(min(5000, len(df))),
    x="total_revenue",
    y="sales_orsy_relevant",
    scatter=False,
    color="red",
    line_kws={"linewidth": 2, "label": "Trendlinie"}
)
plt.legend()
plt.title("Zusammenhang: Gesamtumsatz vs. ORSY-Umsatz")
plt.xlabel("Gesamtumsatz (Summe aller Umsatzarten)")
plt.ylabel("ORSY-Umsatz (sales_orsy_relevant)")
plt.xscale("log")
plt.yscale("log")
plt.tight_layout()

os.makedirs(PLOT_DIR, exist_ok=True)
plt.savefig(os.path.join(PLOT_DIR, "orsy_vs_totalrevenue_scatter.png"))
plt.close()
print("-> Scatterplot gespeichert: eda_plots/orsy_vs_totalrevenue_scatter.png")

# ------------------ Plot 2: Histogramm ORSY-Anteil -------------
plt.figure(figsize=(8,5))
sns.histplot(df["orsy_share"], bins=50, kde=True, color="royalblue")
plt.title("Verteilung des ORSY-Anteils am Gesamtumsatz")
plt.xlabel("ORSY-Anteil (sales_orsy_relevant / Gesamtumsatz)")
plt.ylabel("Anzahl Kunden")
plt.tight_layout()

plt.savefig(os.path.join(PLOT_DIR, "orsy_share_distribution_corrected.png"))
plt.close()
print("-> Histogramm gespeichert: eda_plots/orsy_share_distribution_corrected.png")

print("\n✅ Fertig! Beide Plots liegen im Ordner 'eda_plots/'")