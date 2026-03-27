import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from renewables_forecasting.config.paths import (
    MASTR_WIND_PLANTS_FILTERED_SQLITE,
    MASTR_SOLAR_PLANTS_FILTERED_SQLITE,
    GEONAMES_POSTAL_CODE_DATA,
    ANALYSIS_DIR,
)
from renewables_forecasting.data.mastr import _get_plz_to_lat_lon_mapping


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def coord_quality_analysis(
        db_path,
        table_name: str,
        label: str,
        wind_filter: bool = False,
        out_path: Path | None = None,
):
    wind_where = "WindAnLandOderAufSee = 888" if wind_filter else None
    coords_where = "Breitengrad IS NOT NULL AND Laengengrad IS NOT NULL"
    full_where = f"{coords_where} AND {wind_where}" if wind_where else coords_where

    with sqlite3.connect(db_path) as conn:

        total_in_db = pd.read_sql(f"""
            SELECT COUNT(*) as n FROM {table_name}
            {('WHERE ' + wind_where) if wind_where else ''}
        """, conn).iloc[0]["n"]

        total_with_coords = pd.read_sql(f"""
            SELECT COUNT(*) as n FROM {table_name}
            WHERE {full_where}
        """, conn).iloc[0]["n"]

        df = pd.read_sql(f"""
            SELECT EinheitMastrNummer, Postleitzahl, Breitengrad, Laengengrad, Nettonennleistung
            FROM {table_name}
            WHERE {full_where}
        """, conn)

    df["Postleitzahl"] = df["Postleitzahl"].astype(str).str.zfill(5)

    plz_to_coords = _get_plz_to_lat_lon_mapping(GEONAMES_POSTAL_CODE_DATA)
    df["plz_lat"] = df["Postleitzahl"].map(lambda x: plz_to_coords.get(x, (None, None))[0])
    df["plz_lon"] = df["Postleitzahl"].map(lambda x: plz_to_coords.get(x, (None, None))[1])
    df = df.dropna(subset=["plz_lat", "plz_lon"])

    df["dist_km"] = haversine(
        df["Breitengrad"].values, df["Laengengrad"].values,
        df["plz_lat"].values, df["plz_lon"].values
    )

    total = len(df)
    total_cap_mw = df["Nettonennleistung"].sum() / 1e3
    x_bins = [0, 5, 10, 20, 30, 50, 100, float('inf')]
    bins = ["0-5", "5-10", "10-20", "20-30", "30-50", "50-100", ">100"]

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  {label}")
    lines.append(f"{'='*60}")
    lines.append(f"Total plants in register:                    {total_in_db:,}")
    lines.append(f"Plants with existing MaStR coords:           {total_with_coords:,}  ({total_with_coords/total_in_db*100:.1f}%)")
    lines.append(f"Plants with coords and resolvable PLZ:       {total:,}  ({total/total_with_coords*100:.1f}% of coords)")
    lines.append(f"Total capacity in analysis:                  {total_cap_mw:,.1f} MW\n")
    lines.append(f"{'Bin':>10}  {'Plants':>8}  {'Plants %':>9}  {'Cap (MW)':>10}  {'Cap %':>7}")
    lines.append("-" * 55)

    for i, bin_label in enumerate(bins):
        lo, hi = x_bins[i], x_bins[i + 1]
        mask = ((df["dist_km"] > lo) & (df["dist_km"] <= hi)) if lo > 0 else (df["dist_km"] <= hi)
        count = mask.sum()
        cap = df.loc[mask, "Nettonennleistung"].sum() / 1e3
        lines.append(f"{bin_label:>10}  {count:>8,}  {count/total*100:>8.1f}%  {cap:>10,.1f}  {cap/total_cap_mw*100:>6.1f}%")

    lines.append(f"\nMedian distance: {df['dist_km'].median():.2f} km")
    lines.append(f"Mean distance:   {df['dist_km'].mean():.2f} km")
    lines.append(f"Max distance:    {df['dist_km'].max():.2f} km")

    output = "\n".join(lines)
    print(output)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\n  Saved to: {out_path}")


# ── Solar ─────────────────────────────────────────────────────────────────────
coord_quality_analysis(
    db_path=MASTR_SOLAR_PLANTS_FILTERED_SQLITE,
    table_name="einheiten_solar",
    label="Solar plants",
    wind_filter=False,
    out_path=ANALYSIS_DIR / "solar_coord_quality.txt",
)

# ── Wind ──────────────────────────────────────────────────────────────────────
coord_quality_analysis(
    db_path=MASTR_WIND_PLANTS_FILTERED_SQLITE,
    table_name="einheiten_wind",
    label="Wind plants (onshore only)",
    wind_filter=True,
    out_path=ANALYSIS_DIR / "wind_coord_quality.txt",
)