from renewables_forecasting.config.paths import CYCLICAL_TIME_FEATURES_CSV
from renewables_forecasting.data.time import build_cyclical_time_features


def task_build_cyclical_time_features(
        produces=CYCLICAL_TIME_FEATURES_CSV
):
    build_cyclical_time_features(
        start="2015-01-01 00:00",
        end="2025-12-31 23:00",
        out_path=CYCLICAL_TIME_FEATURES_CSV,
    )
