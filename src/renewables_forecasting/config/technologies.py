# config/technologies.py

from dataclasses import dataclass
from typing import Dict, List

from renewables_forecasting.config.data_sources import WeatherVariableSource, DWD_COSMO_REA6_SOLAR, DWD_COSMO_REA6_WIND, \
    MASTR_SOLAR_VARIABLES, MASTR_WIND_VARIABLES


@dataclass(frozen=True)
class TechnologyConfig:
    name: str
    weather_variables: Dict[str, WeatherVariableSource]
    plant_variables: List[str]


SOLAR = TechnologyConfig(
    name="solar",
    weather_variables=DWD_COSMO_REA6_SOLAR,
    plant_variables=MASTR_SOLAR_VARIABLES
)


WIND = TechnologyConfig(
    name="wind",
    weather_variables=DWD_COSMO_REA6_WIND,
    plant_variables=MASTR_WIND_VARIABLES
)
