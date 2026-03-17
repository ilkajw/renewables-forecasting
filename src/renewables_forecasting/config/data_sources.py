from dataclasses import dataclass

# ---- Copernicus ERA5 ----

ERA5_SOLAR_VARIABLES = {
    "ssrd": "surface_solar_radiation_downwards",
    # "strd": "surface_thermal_radiation_downwards",
    # "t2m": "2m_temperature",
    # "skt": "skin_temperature",
}


ERA5_WIND_VARIABLES = {
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
}


# ---- MaStR ----

MASTR_GESAMTDATENUEBERSICHT_VERSION = "20260308_25.2"
MASTR_ZIP_URL = f"https://download.marktstammdatenregister.de/Gesamtdatenexport_{MASTR_GESAMTDATENUEBERSICHT_VERSION}.zip"

MASTR_SOLAR_VARIABLES = ["NameStromerzeugungseinheit", "EinheitMastrNummer", "Bundesland", "Landkreis", "Gemeinde",
                         "Gemeindeschluessel", "Postleitzahl", "Strasse", "Ort", "Laengengrad", "Breitengrad",
                         "Inbetriebnahmedatum", "DatumEndgueltigeStilllegung", "DatumBeginnVoruebergehendeStilllegung",
                         "DatumWiederaufnahmeBetrieb", "EinheitBetriebsstatus", "Bruttoleistung",
                         "Nettonennleistung", "Einspeisungsart", "Leistungsbegrenzung",
                         "InbetriebnahmedatumAmAktuellenStandort"
                         ]

MASTR_WIND_VARIABLES = ["NameStromerzeugungseinheit", "EinheitMastrNummer", "Bundesland", "Landkreis", "Gemeinde",
                         "Gemeindeschluessel", "Postleitzahl", "Strasse", "Ort", "Laengengrad", "Breitengrad",
                         "Inbetriebnahmedatum", "DatumEndgueltigeStilllegung", "DatumBeginnVoruebergehendeStilllegung",
                         "DatumWiederaufnahmeBetrieb", "EinheitBetriebsstatus", "Kraftwerksnummer", "Bruttoleistung",
                         "Nettonennleistung", "Einspeisungsart", "WindAnLandOderAufSee", "Nabenhoehe",
                         "InbetriebnahmedatumAmAktuellenStandort"
                        ]

# ---- SMARD ----

SMARD_SOLAR_GENERATION_TIMESTAMPS_URL = "https://www.smard.de/app/chart_data/4068/DE/index_hour.json"
SMARD_SOLAR_GENERATION_URL_TEMPLATE = "https://www.smard.de/app/chart_data/4068/DE/4068_DE_hour_{timestamp}.json"


# ---- Geonames ----
GEONAMES_POSTAL_CODES_DATA_URL = "https://download.geonames.org/export/zip/DE.zip"


# ---- DWD COSMO REA6 ----

@dataclass(frozen=True)
class WeatherVariableSource:
    name: str
    base_url: str
    filename_pattern: str


DWD_COSMO_REA6_SOLAR = {
    "ASWDIFD_S": WeatherVariableSource(
        name="ASWDIFD_S",
        base_url="https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/ASWDIFD_S",
        filename_pattern="{var}.2D.{year}{month:02d}.grb.bz2"),

    "ASWDIR_S": WeatherVariableSource(
        name="ASWDIR_S",
        base_url="https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/hourly/2D/ASWDIR_S",
        filename_pattern="{var}.2D.{year}{month:02d}.grb.bz2"),
}

DWD_COSMO_REA6_WIND = {
    "WS_060": WeatherVariableSource(
        name="WS_060",
        base_url="https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_060",
        filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),

    "WS_100": WeatherVariableSource(
        name="WS_100",
        base_url="https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_100",
        filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),

    "WS_125": WeatherVariableSource(
        name="WS_125",
        base_url="https://opendata.dwd.de/climate_environment/REA/COSMO_REA6/converted/hourly/2D/WS_125",
        filename_pattern="{var}m.2D.{year}{month:02d}.nc4"),
}