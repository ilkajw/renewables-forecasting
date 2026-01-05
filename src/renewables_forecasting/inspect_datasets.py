from pathlib import Path
import xarray as xr

from renewables_forecasting.config.paths import DATA_DIR


def inspect_file(path: Path):
    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)

    ds = xr.open_dataset(path, engine="cfgrib" if path.suffix in [".grb", ".grib"] else None)

    print("\n--- DATASET SUMMARY ---")
    print(ds)

    print("\n--- DIMS ---")
    for k, v in ds.dims.items():
        print(f"{k}: {v}")

    print("\n--- VARIABLES ---")
    for name, var in ds.variables.items():
        print(f"\n{name}")
        print(f"  dims: {var.dims}")
        print(f"  shape: {var.shape}")
        if var.attrs:
            print("  attrs:")
            for a, av in var.attrs.items():
                print(f"    {a}: {av}")

    print("\n--- COORDS ---")
    for name, coord in ds.coords.items():
        print(f"\n{name}")
        print(f"  dims: {coord.dims}")
        print(f"  shape: {coord.shape}")
        if coord.attrs:
            print("  attrs:")
            for a, av in coord.attrs.items():
                print(f"    {a}: {av}")

    print("\n--- GLOBAL ATTRS ---")
    for k, v in ds.attrs.items():
        print(f"{k}: {v}")

    print("\n--- GRID MAPPING VARIABLES (important for rotation) ---")
    for name, var in ds.variables.items():
        if "grid_mapping_name" in var.attrs:
            print(f"\n{name}")
            for a, av in var.attrs.items():
                print(f"  {a}: {av}")

    ds.close()


if __name__ == "__main__":

    file_path = DATA_DIR / r"raw\dwd\wind\WS_060\WS_060m.2D.201702.nc4"
    inspect_file(file_path)
