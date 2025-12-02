import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import sys
import os

# Add parent directory to path to import from labs/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import analyse_property_granularity

filename = "../../classification/traffic_accidents.csv"
savefig_path_prefix = "images/granularity/traffic_accidents/traffic_accidents"

traffic_accidents_data: DataFrame = read_csv(filename)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(savefig_path_prefix), exist_ok=True)

# ==========================================
# TEMPORAL GRANULARITY
# ==========================================

# crash_month → Quarter, Semester
def get_month_quarter(month: int) -> str:
    if month <= 3:
        return "Q1"
    elif month <= 6:
        return "Q2"
    elif month <= 9:
        return "Q3"
    else:
        return "Q4"


def get_month_semester(month: int) -> str:
    if month <= 6:
        return "S1"
    else:
        return "S2"


def derive_month(df: DataFrame) -> DataFrame:
    df["Quarter"] = df["crash_month"].apply(get_month_quarter)
    df["Semester"] = df["crash_month"].apply(get_month_semester)
    return df


data_ext: DataFrame = derive_month(traffic_accidents_data)
analyse_property_granularity(data_ext, "crash_month", ["Semester", "Quarter", "crash_month"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_month.png")
plt.close()


# crash_day_of_week - aggregations: weekend/weekday, day

def is_weekend(day: int) -> str:
    # Assumindo: 1=Monday, 7=Sunday
    if day >= 6:  # Saturday(6) or Sunday(7)
        return "Weekend"
    else:
        return "Weekday"


def get_day_name(day: int) -> str:
    days = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 
            5: "Friday", 6: "Saturday", 7: "Sunday"}
    return days.get(day, "Unknown")



def derive_day(df: DataFrame) -> DataFrame:
    df["WeekPart"] = df["crash_day_of_week"].apply(is_weekend)
    df["DayName"] = df["crash_day_of_week"].apply(get_day_name)
    return df


data_ext: DataFrame = derive_day(traffic_accidents_data)
analyse_property_granularity(data_ext, "crash_day_of_week", ["WeekPart", "DayName", "crash_day_of_week"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_day.png")
plt.close()


# crash_hour - aggregations: part of day, hour

def get_hour_period(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:  # 0 <= hour < 6
        return "Night"

def get_hour_category(hour: int) -> str:
    if 7 <= hour < 9:
        return "Morning Rush"
    elif 17 <= hour < 19:
        return "Evening Rush"
    elif 9 <= hour < 17:
        return "Work Hours"
    else:
        return "Off-Peak"

def derive_hour(df: DataFrame) -> DataFrame:
    df["HourPeriod"] = df["crash_hour"].apply(get_hour_period)
    df["HourCategory"] = df["crash_hour"].apply(get_hour_category)
    return df

data_ext: DataFrame = derive_hour(traffic_accidents_data)
analyse_property_granularity(data_ext, "crash_hour", ["HourPeriod", "HourCategory", "crash_hour"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_hour.png")
plt.close()


# ==========================================
# ORDINAL GRANULARITY (NON-TEMPORAL)
# ==========================================

# num_units - aggregations: binary (single/multiple), simple category, detailed category
def get_units_simple(units: int) -> str:
    return "Single" if units == 1 else "Multiple"

def get_units_category(units: int) -> str:
    if units == 1:
        return "Single Vehicle"
    elif units == 2:
        return "Two Vehicles"
    else:
        return "Multi-Vehicle (3+)"

def get_units_detailed(units: int) -> str:
    if units == 1:
        return "1 Vehicle"
    elif units == 2:
        return "2 Vehicles"
    elif units == 3:
        return "3 Vehicles"
    elif units == 4:
        return "4 Vehicles"
    else:
        return "5+ Vehicles"

def derive_num_units(df: DataFrame) -> DataFrame:
    df["UnitsSimple"] = df["num_units"].apply(get_units_simple)
    df["UnitsCategory"] = df["num_units"].apply(get_units_category)
    df["UnitsDetailed"] = df["num_units"].apply(get_units_detailed)
    return df

data_ext = derive_num_units(traffic_accidents_data)
analyse_property_granularity(data_ext, "num_units", ["UnitsSimple", "UnitsCategory", "UnitsDetailed", "num_units"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_num_units.png")
plt.close()


# most_severe_injury - aggregations: binary (serious/non-serious), severity levels, detailed severity
def get_injury_binary(injury: str) -> str:
    injury_upper = str(injury).upper()
    if "FATAL" in injury_upper or "INCAPACITATING" in injury_upper:
        return "Serious Injury"
    else:
        return "Non-Serious Injury"

def get_injury_severity(injury: str) -> str:
    injury_upper = str(injury).upper()
    if "FATAL" in injury_upper:
        return "Fatal"
    elif "INCAPACITATING" in injury_upper:
        return "Severe"
    elif "NON-INCAPACITATING" in injury_upper or "REPORTED" in injury_upper:
        return "Minor"
    else:
        return "No Injury"

def get_injury_detailed(injury: str) -> str:
    injury_upper = str(injury).upper()
    if "FATAL" in injury_upper:
        return "Fatal"
    elif "INCAPACITATING" in injury_upper and "NON" not in injury_upper:
        return "Incapacitating"
    elif "NON-INCAPACITATING" in injury_upper:
        return "Non-Incapacitating"
    elif "REPORTED" in injury_upper:
        return "Reported Not Evident"
    else:
        return "No Indication"

def derive_injury(df: DataFrame) -> DataFrame:
    df["InjuryBinary"] = df["most_severe_injury"].apply(get_injury_binary)
    df["InjurySeverity"] = df["most_severe_injury"].apply(get_injury_severity)
    df["InjuryDetailed"] = df["most_severe_injury"].apply(get_injury_detailed)
    return df

data_ext = derive_injury(traffic_accidents_data)
analyse_property_granularity(data_ext, "most_severe_injury", ["InjuryBinary", "InjurySeverity", "InjuryDetailed", "most_severe_injury"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_injury.png")
plt.close()


# damage - aggregations: binary (significant/non-significant), category (major/moderate/minor), detailed
def get_damage_binary(damage: str) -> str:
    damage_str = str(damage).upper()
    if "$1,500" in damage_str and "OVER" in damage_str:
        return "Significant Damage"
    else:
        return "Non-Significant Damage"

def get_damage_category(damage: str) -> str:
    damage_str = str(damage).upper()
    if "$1,500" in damage_str and "OVER" in damage_str:
        return "Major"
    elif "$501" in damage_str or "1,500" in damage_str:
        return "Moderate"
    else:
        return "Minor"

def get_damage_detailed(damage: str) -> str:
    damage_str = str(damage).upper()
    if "OVER $1,500" in damage_str:
        return "Over $1,500"
    elif "$501" in damage_str and "$1,500" in damage_str:
        return "$501-$1,500"
    elif "$500" in damage_str and "UNDER" in damage_str:
        return "Under $500"
    else:
        return str(damage)

def derive_damage(df: DataFrame) -> DataFrame:
    df["DamageBinary"] = df["damage"].apply(get_damage_binary)
    df["DamageCategory"] = df["damage"].apply(get_damage_category)
    df["DamageDetailed"] = df["damage"].apply(get_damage_detailed)
    return df

data_ext = derive_damage(traffic_accidents_data)
analyse_property_granularity(data_ext, "damage", ["DamageBinary", "DamageCategory", "DamageDetailed", "damage"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_damage.png")
plt.close()


# ==========================================
# CATEGORICAL GRANULARITY
# ==========================================

# weather_condition - has natural order: clear < cloudy < rain < snow
def get_weather_severity(weather: str) -> str:
    weather_upper = str(weather).upper()
    if "SNOW" in weather_upper or "SLEET" in weather_upper or "FREEZING" in weather_upper:
        return "Severe Weather"
    elif "RAIN" in weather_upper or "FOG" in weather_upper or "BLOWING" in weather_upper:
        return "Adverse Weather"
    elif "CLOUDY" in weather_upper or "OVERCAST" in weather_upper:
        return "Cloudy Weather"
    else:
        return "Clear Weather"

def get_weather_binary(weather: str) -> str:
    weather_upper = str(weather).upper()
    if "CLEAR" in weather_upper or "UNKNOWN" in weather_upper:
        return "Clear/Unknown"
    else:
        return "Adverse Conditions"

def derive_weather(df: DataFrame) -> DataFrame:
    df["WeatherBinary"] = df["weather_condition"].apply(get_weather_binary)
    df["WeatherSeverity"] = df["weather_condition"].apply(get_weather_severity)
    return df

data_ext = derive_weather(traffic_accidents_data)
analyse_property_granularity(data_ext, "weather_condition", ["WeatherBinary", "WeatherSeverity", "weather_condition"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_weather.png")
plt.close()


# lighting_condition - has natural order: daylight > dusk > dawn > darkness lighted > darkness
def get_lighting_quality(lighting: str) -> str:
    lighting_upper = str(lighting).upper()
    if "DAYLIGHT" in lighting_upper:
        return "Good Lighting"
    elif "DUSK" in lighting_upper or "DAWN" in lighting_upper:
        return "Moderate Lighting"
    elif "LIGHTED" in lighting_upper:
        return "Artificial Lighting"
    else:
        return "Poor Lighting"

def get_lighting_binary(lighting: str) -> str:
    lighting_upper = str(lighting).upper()
    if "DAYLIGHT" in lighting_upper:
        return "Daylight"
    else:
        return "Non-Daylight"

def derive_lighting(df: DataFrame) -> DataFrame:
    df["LightingBinary"] = df["lighting_condition"].apply(get_lighting_binary)
    df["LightingQuality"] = df["lighting_condition"].apply(get_lighting_quality)
    return df

data_ext = derive_lighting(traffic_accidents_data)
analyse_property_granularity(data_ext, "lighting_condition", ["LightingBinary", "LightingQuality", "lighting_condition"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_lighting.png")
plt.close()


# roadway_surface_cond - has natural order: dry > wet > snow/slush > ice
def get_surface_safety(surface: str) -> str:
    surface_upper = str(surface).upper()
    if "ICE" in surface_upper:
        return "Hazardous"
    elif "SNOW" in surface_upper or "SLUSH" in surface_upper:
        return "Poor"
    elif "WET" in surface_upper:
        return "Fair"
    else:
        return "Good"

def get_surface_binary(surface: str) -> str:
    surface_upper = str(surface).upper()
    if "DRY" in surface_upper or "UNKNOWN" in surface_upper:
        return "Dry/Unknown"
    else:
        return "Compromised Surface"

def derive_surface(df: DataFrame) -> DataFrame:
    df["SurfaceBinary"] = df["roadway_surface_cond"].apply(get_surface_binary)
    df["SurfaceSafety"] = df["roadway_surface_cond"].apply(get_surface_safety)
    return df

data_ext = derive_surface(traffic_accidents_data)
analyse_property_granularity(data_ext, "roadway_surface_cond", ["SurfaceBinary", "SurfaceSafety", "roadway_surface_cond"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_surface.png")
plt.close()


# traffic_control_device - group by control type
def get_control_type(control: str) -> str:
    control_upper = str(control).upper()
    if "TRAFFIC SIGNAL" in control_upper:
        return "Traffic Signal"
    elif "STOP" in control_upper or "FLASHER" in control_upper:
        return "Stop Sign/Flasher"
    elif "NO CONTROLS" in control_upper:
        return "No Controls"
    else:
        return "Other Controls"

def get_control_binary(control: str) -> str:
    control_upper = str(control).upper()
    if "NO CONTROLS" in control_upper or "UNKNOWN" in control_upper:
        return "Uncontrolled"
    else:
        return "Controlled"

def derive_control(df: DataFrame) -> DataFrame:
    df["ControlBinary"] = df["traffic_control_device"].apply(get_control_binary)
    df["ControlType"] = df["traffic_control_device"].apply(get_control_type)
    return df

data_ext = derive_control(traffic_accidents_data)
analyse_property_granularity(data_ext, "traffic_control_device", ["ControlBinary", "ControlType", "traffic_control_device"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_control.png")
plt.close()


# first_crash_type - group by collision pattern
def get_crash_pattern(crash_type: str) -> str:
    crash_upper = str(crash_type).upper()
    if "REAR END" in crash_upper or "REAR TO" in crash_upper:
        return "Rear-End"
    elif "ANGLE" in crash_upper or "HEAD ON" in crash_upper:
        return "Intersection/Head-On"
    elif "SIDESWIPE" in crash_upper:
        return "Sideswipe"
    elif "TURNING" in crash_upper:
        return "Turning"
    elif "FIXED OBJECT" in crash_upper or "PARKED" in crash_upper:
        return "Fixed Object"
    elif "PEDESTRIAN" in crash_upper or "PEDALCYCLIST" in crash_upper or "ANIMAL" in crash_upper:
        return "Non-Vehicle"
    else:
        return "Other"

def get_crash_complexity(crash_type: str) -> str:
    crash_upper = str(crash_type).upper()
    if "REAR END" in crash_upper or "FIXED OBJECT" in crash_upper or "PARKED" in crash_upper:
        return "Simple Collision"
    else:
        return "Complex Collision"

def derive_crash_type(df: DataFrame) -> DataFrame:
    df["CrashComplexity"] = df["first_crash_type"].apply(get_crash_complexity)
    df["CrashPattern"] = df["first_crash_type"].apply(get_crash_pattern)
    return df

data_ext = derive_crash_type(traffic_accidents_data)
analyse_property_granularity(data_ext, "first_crash_type", ["CrashComplexity", "CrashPattern", "first_crash_type"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_crash_type.png")
plt.close()


# trafficway_type - group by complexity
def get_trafficway_complexity(trafficway: str) -> str:
    trafficway_upper = str(trafficway).upper()
    if "DIVIDED" in trafficway_upper or "MEDIAN" in trafficway_upper:
        return "Divided"
    elif "ONE-WAY" in trafficway_upper or "RAMP" in trafficway_upper:
        return "One-Way/Ramp"
    elif "NOT DIVIDED" in trafficway_upper:
        return "Not Divided"
    else:
        return "Other/Unknown"

def get_trafficway_binary(trafficway: str) -> str:
    trafficway_upper = str(trafficway).upper()
    if "DIVIDED" in trafficway_upper:
        return "Divided"
    else:
        return "Not Divided"

def derive_trafficway(df: DataFrame) -> DataFrame:
    df["TrafficwayBinary"] = df["trafficway_type"].apply(get_trafficway_binary)
    df["TrafficwayComplexity"] = df["trafficway_type"].apply(get_trafficway_complexity)
    return df

data_ext = derive_trafficway(traffic_accidents_data)
analyse_property_granularity(data_ext, "trafficway_type", ["TrafficwayBinary", "TrafficwayComplexity", "trafficway_type"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_trafficway.png")
plt.close()


# road_defect - has natural order: no defects > minor defects > major defects
def get_defect_severity(defect: str) -> str:
    defect_upper = str(defect).upper()
    if "NO DEFECTS" in defect_upper:
        return "No Defects"
    elif "UNKNOWN" in defect_upper:
        return "Unknown"
    else:
        return "Has Defects"

def derive_road_defect(df: DataFrame) -> DataFrame:
    df["DefectPresence"] = df["road_defect"].apply(get_defect_severity)
    return df

data_ext = derive_road_defect(traffic_accidents_data)
analyse_property_granularity(data_ext, "road_defect", ["DefectPresence", "road_defect"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_defect.png")
plt.close()

print("Granularity analysis completed!")