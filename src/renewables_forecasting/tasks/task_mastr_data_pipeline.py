from datetime import date

from renewables_forecasting.config.paths import MASTR_GESAMTDATENUEBERSICHT_PATH, MASTR_SOLAR_PLANTS_CSV_PATH, \
    MASTR_SOLAR_PLANTS_SQLITE_PATH, MASTR_WIND_PLANTS_CSV_PATH, MASTR_WIND_PLANTS_SQLITE_PATH
from renewables_forecasting.config.data_sources import MASTR_ZIP_URL, MASTR_SOLAR_VARIABLES

from renewables_forecasting.data.mastr import download_mastr_gesamtdatenuebersicht, \
    filter_xmls_from_gesamtdatenuebersicht_to_csv, csv_to_sql


def task_download_mastr_gesamtdatenuebersicht(produces=MASTR_GESAMTDATENUEBERSICHT_PATH):
    # Pytask automatically creates parent directories of files in 'produces'

    download_mastr_gesamtdatenuebersicht(
        url=MASTR_ZIP_URL,
        out_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        overwrite=False
    )


def filter_solar_xmls_from_gesamtdatenuebersicht_to_csv(
        depends_on=MASTR_GESAMTDATENUEBERSICHT_PATH,
        produces=MASTR_SOLAR_PLANTS_CSV_PATH
):

    filter_xmls_from_gesamtdatenuebersicht_to_csv(
        zip_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        naming_files="EinheitenSolar",
        naming_units="EinheitSolar",
        start=date(2015, 1, 1),
        end=date(2025, 12, 31),  # inclusive
        variables=MASTR_SOLAR_VARIABLES,
        exclude_filters={
            "EinheitBetriebstatus": "31"  # filter out plants still 'InPlanung'
        },
        out_csv=MASTR_SOLAR_PLANTS_CSV_PATH
    )


def task_solar_csv_to_sqlite(depends_on=MASTR_SOLAR_PLANTS_CSV_PATH, produces=MASTR_SOLAR_PLANTS_SQLITE_PATH):
    csv_to_sql(
        csv_path=MASTR_SOLAR_PLANTS_CSV_PATH,
        sql_path=MASTR_SOLAR_PLANTS_SQLITE_PATH
    )


"""
def task_filter_wind_xml_to_csv(depends_on=MASTR_GESAMTDATENUEBERSICHT_PATH, produces=MASTR_WIND_PLANTS_CSV_PATH):


    filter_xmls_from_gesamtdatenuebersicht_to_csv(
        zip_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        naming_files="EinheitenWind",
        naming_units="EinheitWind",
        start=date(2015, 1, 1),
        end=date(2025, 12, 31),  # inclusive
        variables=MASTR_WIND_VARIABLES,
        exclude_filters={
            "EinheitBetriebstatus": "31"  # filter out plants still 'InPlanung'
        },
        out_csv=MASTR_WIND_PLANTS_CSV_PATH
    )


def task_wind_csv_to_sqlite(depends_on=MASTR_WIND_PLANTS_CSV_PATH, produces=MASTR_WIND_PLANTS_SQLITE_PATH):

    csv_to_sql(
        csv_path=MASTR_WIND_PLANTS_CSV_PATH,
        sql_path=MASTR_WIND_PLANTS_SQLITE_PATH
    )
"""
