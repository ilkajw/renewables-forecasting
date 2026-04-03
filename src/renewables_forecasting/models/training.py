from renewables_forecasting.config.paths import SOLAR_FEATURES_DIR, PROJECT_ROOT
from renewables_forecasting.models.train import train

CONFIG_PATH = PROJECT_ROOT / "src" / "renewables_forecasting" / "config" / "solar_model_config.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "models" / "solar" / "solar_v5"

train(
    features_dir=SOLAR_FEATURES_DIR,
    config_path=CONFIG_PATH,
    out_dir=OUT_DIR,
    time_col="time",
    value_col="generation_mwh",
    early_stopping_patience=30,
    num_workers=0,
)
