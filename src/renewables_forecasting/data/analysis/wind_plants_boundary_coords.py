import pandas as pd

from renewables_forecasting.config.paths import MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV, ANALYSIS_DIR

df = pd.read_csv(MASTR_WIND_PLANTS_EFFECTIVE_START_WITH_COORDS_CSV, dtype={"Postleitzahl": str})

df = df.dropna(subset=["Breitengrad", "Laengengrad"])

lines = []
lines.append("=" * 50)
lines.append("  Wind plant coordinate bounds")
lines.append("=" * 50)
lines.append(f"Total plants with coords: {len(df):,}")
lines.append(f"\nLatitude  — min: {df['Breitengrad'].min():.4f}  max: {df['Breitengrad'].max():.4f}")
lines.append(f"Longitude — min: {df['Laengengrad'].min():.4f}  max: {df['Laengengrad'].max():.4f}")

output = "\n".join(lines)
print(output)

out_path = ANALYSIS_DIR / "wind_plant_coord_bounds.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output, encoding="utf-8")
print(f"\nSaved to: {out_path}")