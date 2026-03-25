import xarray as xr
import pandas as pd
from pathlib import Path

from renewables_forecasting.config.paths import (
    SOLAR_CAPACITY_GRIDS_ZARR_STORE,
    WIND_CAPACITY_GRIDS_ZARR_STORE,
    ANALYSIS_DIR,
)


def total_capacity_end_of_year(
        store_dir: Path,
        years: range,
) -> pd.DataFrame:
    rows = []
    for year in years:
        store = store_dir / f"capacity_{year}_12.zarr"
        if not store.exists():
            print(f"  Missing: {store.name}")
            continue

        ds = xr.open_dataset(store, engine="zarr")
        da = ds["capacity_kw"].sel(time=f"{year}-12-31")
        total_mw = float(da.sum()) / 1e3
        rows.append({"year": year, "total_mw": round(total_mw, 1)})
        ds.close()

    return pd.DataFrame(rows)


years = range(2015, 2026)

solar_df = total_capacity_end_of_year(SOLAR_CAPACITY_GRIDS_ZARR_STORE, years)
wind_df = total_capacity_end_of_year(WIND_CAPACITY_GRIDS_ZARR_STORE, years)

lines = []
lines.append("=" * 40)
lines.append("  Total capacity end of year (MW)")
lines.append("=" * 40)
lines.append(f"\n{'Year':>6}  {'Solar (MW)':>12}  {'Wind (MW)':>12}")
lines.append("-" * 36)

for year in years:
    solar_row = solar_df[solar_df["year"] == year]
    wind_row = wind_df[wind_df["year"] == year]
    solar_mw = f"{solar_row['total_mw'].iloc[0]:,.1f}" if not solar_row.empty else "N/A"
    wind_mw = f"{wind_row['total_mw'].iloc[0]:,.1f}" if not wind_row.empty else "N/A"
    lines.append(f"{year:>6}  {solar_mw:>12}  {wind_mw:>12}")

output = "\n".join(lines)
print(output)

out_path = ANALYSIS_DIR / "capacity_end_of_year.txt"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(output, encoding="utf-8")
print(f"\nSaved to: {out_path}")