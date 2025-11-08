import pandas as pd
import os

# === Pfad zur CSV ===
data_path = os.path.join("data", "dataset_wuerth.csv")

# === Daten laden ===
df = pd.read_csv(data_path)

print("\n=========== REGION, DISTRICT & BRANCH OFFICE ANALYSE ===========\n")

# === 1️⃣ Anzahl einzigartiger Regionen ===
unique_regions = df["region"].nunique()
print(f"Anzahl Regionen: {unique_regions}")
print("Regionen:", sorted(df["region"].unique()))

# === 2️⃣ Anzahl einzigartiger Districts ===
unique_districts = df["district"].nunique()
print(f"\nAnzahl Districts insgesamt: {unique_districts}")

# === 3️⃣ Anzahl Districts pro Region ===
districts_per_region = df.groupby("region")["district"].nunique().reset_index()
districts_per_region.columns = ["region", "districts_count"]
print("\nDistricts pro Region:")
print(districts_per_region.to_string(index=False))

# === 4️⃣ Anzahl Branch Offices pro Region ===
# Branch Offices sind evtl. mit NaN, die filtern wir raus
branch_per_region = (
    df.dropna(subset=["branch_office"])
    .groupby("region")["branch_office"]
    .nunique()
    .reset_index()
)
branch_per_region.columns = ["region", "branch_office_count"]
print("\nBranch Offices pro Region:")
print(branch_per_region.to_string(index=False))

# === 5️⃣ Kunden pro Region / District / Branch (optional) ===
customers_per_region = df["region"].value_counts().sort_index()
customers_per_district = df["district"].value_counts().head(10)
customers_per_branch = df["branch_office"].value_counts().head(10)

print("\nKunden pro Region:")
print(customers_per_region)

print("\nTop 10 Districts nach Kundenzahl:")
print(customers_per_district)

print("\nTop 10 Branch Offices nach Kundenzahl:")
print(customers_per_branch)

# === 6️⃣ CSV-Export der Übersicht ===
summary_path = os.path.join("data", "geo_structure_summary.csv")
districts_per_region.merge(branch_per_region, on="region").to_csv(summary_path, index=False)
print(f"\n✅ Strukturübersicht gespeichert unter: {summary_path}")