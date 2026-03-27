import sqlite3
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd

from renewables_forecasting.config.paths import MASTR_WIND_PLANTS_FILTERED_SQLITE, PLOTS_DIR, ANALYSIS_DIR

outdir = PLOTS_DIR / "wind"
outdir.mkdir(parents=True, exist_ok=True)

annotations = False

with sqlite3.connect(MASTR_WIND_PLANTS_FILTERED_SQLITE) as conn:
    df = pd.read_sql("""
        SELECT Nabenhoehe
        FROM einheiten_wind
        WHERE Nabenhoehe IS NOT NULL
    """, conn)

df["Nabenhoehe"] = pd.to_numeric(df["Nabenhoehe"], errors="coerce")
df = df.dropna()

print(f"Wind plants with hub height data: {len(df):,}")
print(f"Min:    {df['Nabenhoehe'].min():.1f} m")
print(f"Median: {df['Nabenhoehe'].median():.1f} m")
print(f"Mean:   {df['Nabenhoehe'].mean():.1f} m")
print(f"Max:    {df['Nabenhoehe'].max():.1f} m")

fig, ax = plt.subplots(figsize=(6, 8))
ax.boxplot(df["Nabenhoehe"], vert=True, patch_artist=True,
           boxprops=dict(facecolor="steelblue", alpha=0.7))
ax.set_ylabel("Hub height (m)")
ax.set_xticks([])
# ax.set_title("Onshore wind turbine hub heights (MaStR)")

# Annotate key statistics
q25 = df["Nabenhoehe"].quantile(0.25)
median = df["Nabenhoehe"].median()
q75 = df["Nabenhoehe"].quantile(0.75)
mean = df["Nabenhoehe"].mean()

if annotations:
    for val, label, offset in [
        (df["Nabenhoehe"].min(), f"Min: {df['Nabenhoehe'].min():.0f} m", 0),
        (q25, f"Q25: {q25:.0f} m", 0),
        (median, f"Median: {median:.0f} m", 0),
        (mean, f"Mean: {mean:.0f} m", 8),  # offset mean down by 8m
        (q75, f"Q75: {q75:.0f} m", 0),
        (df["Nabenhoehe"].max(), f"Max: {df['Nabenhoehe'].max():.0f} m", 0),
    ]:
        ax.annotate(label, xy=(1, val), xytext=(1.15, val - offset),
                    va="center", fontsize=9)

plt.tight_layout()
plt.savefig(outdir / "wind_hub_heights_boxplot.png", dpi=150, bbox_inches="tight")

lines = []
lines.append("=" * 40)
lines.append("  Wind turbine hub height statistics")
lines.append("=" * 40)
lines.append(f"Plants with hub height data: {len(df):,}")
lines.append(f"Min:    {df['Nabenhoehe'].min():.1f} m")
lines.append(f"Q25:    {q25:.1f} m")
lines.append(f"Median: {median:.1f} m")
lines.append(f"Mean:   {mean:.1f} m")
lines.append(f"Q75:    {q75:.1f} m")
lines.append(f"Max:    {df['Nabenhoehe'].max():.1f} m")

output = "\n".join(lines)
print(output)

out_path = ANALYSIS_DIR / "wind_hub_height_stats.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output, encoding="utf-8")
print(f"\nSaved to: {out_path}")

plt.show()
