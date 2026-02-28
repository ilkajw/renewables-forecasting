# config/technologies.py
from dataclasses import dataclass
from typing import Dict, Optional, List
from pathlib import Path

from renewables_forecasting.config.paths import RAW_DATA_DIR, DATA_DIR


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
    weather_variables: Dict[str, VariableConfig]
    plant_variables: List[str]
    plant_data_raw_subdir: Path

# todo:
#  fix subdir definitions and naming: rename raw subdir to raw weather subdir,
#  define subdir for raw mastr solar/wind data.
#  move paths and source definitions to paths.py and data_sources.py.
#  check usage of restructured definitions


SOLAR = TechnologyConfig(
    name="solar",
    raw_subdir=DATA_DIR / "raw/dwd/solar",
    plant_data_raw_subdir=RAW_DATA_DIR / "mastr/solar",
    var_zarr_store=DATA_DIR / "processed/dwd/solar",
    feature_zarr_store=DATA_DIR / "features/dwd/solar",
    weather_variables={
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
    plant_variables=["NameStromerzeugungseinheit", "EinheitMastrNummer", "Bundesland", "Landkreis", "Gemeinde",
                     "Gemeindeschluessel", "Postleitzahl", "Strasse", "Ort", "Laengengrad", "Breitengrad",
                     "Inbetriebnahmedatum", "DatumEndgueltigeStilllegung", "DatumBeginnVoruebergehendeStilllegung",
                     "DatumWiederaufnahmeBetrieb", "EinheitBetriebsstatus", "Bruttoleistung", "Nettonennleistung",
                     "Einspeisungsart", "Leistungsbegrenzung", "InbetriebnahmedatumAmAktuellenStandort"]
)


WIND = TechnologyConfig(
    name="wind",
    raw_subdir=DATA_DIR / "raw/dwd/wind",
    var_zarr_store=None,
    feature_zarr_store=DATA_DIR / "features/dwd/wind",
    weather_variables={
        "WS_060": VariableConfig(
            name="WS_060",
            base_url="https://opendata.dwd.de/climate_environment/REA/"
                     "COSMO_REA6/converted/hourly/2D/WS_060",
            filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),

        "WS_100": VariableConfig(
            name="WS_100",
            base_url="https://opendata.dwd.de/climate_environment/REA/"
                     "COSMO_REA6/converted/hourly/2D/WS_100",
            filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),

        "WS_125": VariableConfig(
            name="WS_125",
            base_url="https://opendata.dwd.de/climate_environment/REA/"
                     "COSMO_REA6/converted/hourly/2D/WS_125",
            filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),
    },
    plant_variables=[],
    plant_data_raw_subdir=RAW_DATA_DIR / "mastr" / "wind"

)
