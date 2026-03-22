from renewables_forecasting.config.paths import MASTR_GESAMTDATENUEBERSICHT_PATH, GEONAMES_POSTAL_CODE_DATA
from renewables_forecasting.config.data_sources import MASTR_ZIP_URL, GEONAMES_POSTAL_CODES_DATA_URL
from renewables_forecasting.data.mastr import download_mastr_gesamtdatenuebersicht, download_geonames_postal_code_data


# ── Download the global Gesamtdatenuebersicht with all power plants from MaStR ─────────────────────

def task_download_mastr_gesamtdatenuebersicht(
        produces=MASTR_GESAMTDATENUEBERSICHT_PATH
):

    download_mastr_gesamtdatenuebersicht(
        url=MASTR_ZIP_URL,
        out_path=MASTR_GESAMTDATENUEBERSICHT_PATH,
        overwrite=False
    )


# ── Download postal code data from Geonames for coordinate assignment ──────────────────────────────
def task_download_geonames_postal_code_data(
        produces=GEONAMES_POSTAL_CODE_DATA
):

    download_geonames_postal_code_data(
        url=GEONAMES_POSTAL_CODES_DATA_URL,
        out_path=GEONAMES_POSTAL_CODE_DATA,
    )