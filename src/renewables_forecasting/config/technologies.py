# config/technologies.py
from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

from renewables_forecasting.config.paths import DATA_DIR


@dataclass(frozen=True)
class VariableConfig:
    name: str
    base_url: str
    filename_pattern: str


@dataclass(frozen=True)
class TechnologyConfig:
    name: str
    raw_subdir: Path
    var_zarr_store: Optional[Path]
    feature_zarr_store: Path
    variables: Dict[str, VariableConfig]


SOLAR = TechnologyConfig(
    name="solar",
    raw_subdir=DATA_DIR / "raw/dwd/solar",
    var_zarr_store=DATA_DIR / "processed/dwd/solar",
    feature_zarr_store=DATA_DIR / "features/dwd/solar",
    variables={
        "ASWDIFD_S": VariableConfig(
            name="ASWDIFD_S",
            base_url="https://opendata.dwd.de/"
            "climate_environment/REA/COSMO_REA6/hourly/2D/ASWDIFD_S",
            filename_pattern="{var}.2D.{year}{month:02d}.grb.bz2"),

        "ASWDIR_S": VariableConfig(
            name="ASWDIR_S",
            base_url="https://opendata.dwd.de/"
            "climate_environment/REA/COSMO_REA6/hourly/2D/ASWDIR_S",
            filename_pattern="{var}.2D.{year}{month:02d}.grb.bz2"),
    },
)


WIND = TechnologyConfig(
    name="wind",
    raw_subdir=DATA_DIR / "raw/dwd/wind",
    var_zarr_store=None,
    feature_zarr_store=DATA_DIR / "features/dwd/wind",
    variables={
        "WD_060": VariableConfig(
            name="WD_060",
            base_url="https://opendata.dwd.de/climate_environment/REA/"
                     "COSMO_REA6/converted/hourly/2D/WD_060",
            filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),

        "WD_100": VariableConfig(
            name="WD_100",
            base_url="https://opendata.dwd.de/climate_environment/REA/"
                     "COSMO_REA6/converted/hourly/2D/WD_100",
            filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),

        "WD_150": VariableConfig(
            name="WD_150",
            base_url="https://opendata.dwd.de/climate_environment/REA/"
                     "COSMO_REA6/converted/hourly/2D/WD_150",
            filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),
    },
)
