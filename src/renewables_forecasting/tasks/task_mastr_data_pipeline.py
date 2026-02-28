from datetime import date

from renewables_forecasting.config.technologies import SOLAR
from renewables_forecasting.data.mastr import download_mastr_gesamtdatenuebersicht, \
    filter_solar_xml_from_gesamtdatenuebersicht_to_csv, csv_to_sql
from renewables_forecasting.config.paths import RAW_DATA_DIR, MASTR_GESAMTDATENUEBERSICHT_PATH, \
    MASTR_SOLAR_PLANTS_CSV_PATH, MASTR_SOLAR_PLANTS_SQLITE_PATH
from renewables_forecasting.config.data_sources import MASTR_ZIP_URL


def task_download_mastr_gesamtdatenuebersicht(
        produces=MASTR_GESAMTDATENUEBERSICHT_PATH,
):

    # Pytask automatically creates parent directories of files in 'produces'

    download_mastr_gesamtdatenuebersicht(
        url=MASTR_ZIP_URL,
        target_dir=RAW_DATA_DIR,
        overwrite=False
    )


def task_filter_solar_xml_to_csv(
        depends_on=MASTR_GESAMTDATENUEBERSICHT_PATH,
        produces=MASTR_SOLAR_PLANTS_CSV_PATH,
):

    # Pytask automatically creates parent directories of files in 'produces'

    filter_solar_xml_from_gesamtdatenuebersicht_to_csv(
        zip_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        inbetriebnahme_start=date(2010, 1, 1),  # todo: check again from 1-1-2015
        inbetriebnahme_end=date(2025, 12, 31),
        variables=SOLAR.plant_variables,
        out_csv=MASTR_SOLAR_PLANTS_CSV_PATH
    )


def task_solar_csv_to_sqlite(
        depends_on=MASTR_SOLAR_PLANTS_CSV_PATH,
        produces=MASTR_SOLAR_PLANTS_SQLITE_PATH
):

    csv_to_sql(
        csv_path=MASTR_SOLAR_PLANTS_CSV_PATH,
        sql_path=MASTR_SOLAR_PLANTS_SQLITE_PATH
    )
