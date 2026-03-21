
# Bounding box of Germany
# Value definitions consider 0.125 degrees grid cell width in each direction from centroid

# Exact values of border are 5.866333, 15.041917 west- and east-most longitudes.
# Exact values of border are 47.270111, 55.058639 north- and south-most latitudes
GERMANY_LON_MIN, GERMANY_LON_MAX = 5.75, 15.00
GERMANY_LAT_MIN, GERMANY_LAT_MAX = 47.25, 55.00

# Bounding box of Europe
EUROPE_LON_MIN, EUROPE_LON_MAX = -25.0, 45.0
EUROPE_LAT_MIN, EUROPE_LAT_MAX = 34.5, 71.5

# German local timezone — handles CET/CEST transitions automatically
GERMAN_TZ = "Europe/Berlin"

# PLZs missing in Geonames Postal Code data. They were looked up manually on Google Maps
MISSING_PLZS_TO_COORDS_DICT = {
    "04861": (51.509, 13.003),  # Torgau
    "06485": (51.713, 11.130),  # Quedlinburg
    "06711": (51.096, 12.116),  # Zeitz
    "06772": (51.730, 12.378),  # Gräfenhainichen
    "09434": (50.736, 13.105),  # Krumhermersdorf
    "22961": (53.675, 10.342),  # Oetjendorf
    "25867": (54.682, 8.702),   # Oland
    "31035": (52.088, 9.858),   # Despetal
    "33333": (51.885, 8.444),   # Gütersloh
    "39628": (52.651, 11.741),  # Bismark
    "64759": (49.547, 8.991),   # Sensbachtal. Gemeinde has been integrated into Oberzent, therefor gets its coords
    "64760": (49.547, 8.991),   # Oberzent
    "82475": (47.416, 10.978),  # Schneefernerhaus
    "98694": (50.626, 10.935),  # Ilmenau
    "99090": (51.009, 10.925),  # Erfurt
    "99095": (51.052, 11.040),  # Erfurt
    "99331": (50.712, 10.829),  # Geratal
}


# DWD

GRID_RESOLUTION_M = 30000.0