"""The variables and levels of the WeatherBench headline scores."""

VAR_MAP = {
    "geopotential": "Z",
    "temperature": "T",
    "u_component_of_wind": "U",
    "v_component_of_wind": "V",
    "specific_humidity": "Q",
    "2m_temperature": "T2M",
    "mean_sea_level_pressure": "SP",
    "10m_u_component_of_wind": "U10M",
    "10m_v_component_of_wind": "V10M",
}

# Variables and their primary evaluation levels (None for surface variables)
VARIABLE_LEVELS = [
    ("geopotential", 500),
    ("temperature", 850),
    ("specific_humidity", 700),
    ("u_component_of_wind", 850),
    ("v_component_of_wind", 850),
    ("2m_temperature", None),
    ("mean_sea_level_pressure", None),
    ("10m_u_component_of_wind", None),
    ("10m_v_component_of_wind", None),
]

HEADLINE_VARIABLES = sorted({var for var, _ in VARIABLE_LEVELS})
HEADLINE_LEVELS = sorted(
    {level for _, level in VARIABLE_LEVELS if level is not None}
)
