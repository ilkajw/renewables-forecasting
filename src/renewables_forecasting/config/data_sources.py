from dataclasses import dataclass


# ---- DWD weather data sources ----

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

# ---- MaStR plant data sources ----

MASTR_VERSION_NAME = "Gesamtdatenexport_20260101_25.2.zip"

MASTR_ZIP_URL = f"https://download.marktstammdatenregister.de/Stichtag/{MASTR_VERSION_NAME}"

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
# ---- SMARD electricity generation data sources ----

SMARD_SOLAR_GENERATION_TIMESTAMPS_URL = "https://www.smard.de/app/chart_data/4068/DE/index_hour.json"

SMARD_SOLAR_GENERATION_URL_TEMPLATE = "https://www.smard.de/app/chart_data/4068/DE/4068_DE_hour_{timestamp}.json"
