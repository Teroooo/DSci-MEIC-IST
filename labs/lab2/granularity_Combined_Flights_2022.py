import matplotlib.pyplot as plt
from pandas import read_csv, DataFrame
import sys
import os

# Add parent directory to path to import from labs/utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.dslabs_functions import analyse_property_granularity

filename = "../../classification/Combined_Flights_2022.csv"
savefig_path_prefix = "images/granularity/Combined_Flights_2022/Combined_Flights_2022"

print("Reading CSV")

Combined_Flights_data: DataFrame = read_csv(filename)

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(savefig_path_prefix), exist_ok=True)

# ------------------
# TEMPORAL GRANULARITY
# ------------------

# Month → Quarter, Semester
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
    df["Quarter"] = df["Month"].apply(get_month_quarter)
    df["Semester"] = df["Month"].apply(get_month_semester)
    return df

print("Analyzing Month granularity...")
data_ext: DataFrame = derive_month(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "Month", ["Semester", "Quarter", "Month"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_month.png")
plt.close()


# DayofMonth → MonthPart, WeekOfMonth
def get_week_of_month(day: int) -> str:
    if day <= 7:
        return "Week 1 (1-7)"
    elif day <= 14:
        return "Week 2 (8-14)"
    elif day <= 21:
        return "Week 3 (15-21)"
    else:
        return "Week 4+ (22-31)"

def get_month_part(day: int) -> str:
    if day <= 10:
        return "Beginning (1-10)"
    elif day <= 20:
        return "Middle (11-20)"
    else:
        return "End (21-31)"

def derive_day_of_month(df: DataFrame) -> DataFrame:
    df["MonthPart"] = df["DayofMonth"].apply(get_month_part)
    df["WeekOfMonth"] = df["DayofMonth"].apply(get_week_of_month)
    return df

print("Analyzing DayofMonth granularity...")
data_ext: DataFrame = derive_day_of_month(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "DayofMonth", ["MonthPart", "WeekOfMonth", "DayofMonth"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_dayofmonth.png")
plt.close()


# DayOfWeek → WeekPart, DayName
def is_weekend(day: int) -> str:
    # DayOfWeek: 1=Monday, 7=Sunday
    if day >= 6:  # Saturday(6) or Sunday(7)
        return "Weekend"
    else:
        return "Weekday"

def get_day_name(day: int) -> str:
    days = {1: "Monday", 2: "Tuesday", 3: "Wednesday", 4: "Thursday", 
            5: "Friday", 6: "Saturday", 7: "Sunday"}
    return days.get(day, "Unknown")

def derive_day(df: DataFrame) -> DataFrame:
    df["WeekPart"] = df["DayOfWeek"].apply(is_weekend)
    df["DayName"] = df["DayOfWeek"].apply(get_day_name)
    return df

print("Analyzing DayOfWeek granularity...")
data_ext: DataFrame = derive_day(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "DayOfWeek", ["WeekPart", "DayName", "DayOfWeek"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_day.png")
plt.close()




# ------------------
# GEOGRAPHIC GRANULARITY
# ------------------

# US Regions mapping
US_REGIONS = {
    # Northeast
    'CT': 'Northeast', 'ME': 'Northeast', 'MA': 'Northeast', 'NH': 'Northeast',
    'RI': 'Northeast', 'VT': 'Northeast', 'NJ': 'Northeast', 'NY': 'Northeast',
    'PA': 'Northeast',
    # Midwest
    'IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest', 'OH': 'Midwest',
    'WI': 'Midwest', 'IA': 'Midwest', 'KS': 'Midwest', 'MN': 'Midwest',
    'MO': 'Midwest', 'NE': 'Midwest', 'ND': 'Midwest', 'SD': 'Midwest',
    # South
    'DE': 'South', 'FL': 'South', 'GA': 'South', 'MD': 'South',
    'NC': 'South', 'SC': 'South', 'VA': 'South', 'WV': 'South',
    'AL': 'South', 'KY': 'South', 'MS': 'South', 'TN': 'South',
    'AR': 'South', 'LA': 'South', 'OK': 'South', 'TX': 'South',
    # West
    'AZ': 'West', 'CO': 'West', 'ID': 'West', 'MT': 'West',
    'NV': 'West', 'NM': 'West', 'UT': 'West', 'WY': 'West',
    'AK': 'West', 'CA': 'West', 'HI': 'West', 'OR': 'West', 'WA': 'West',
    # Territories
    'PR': 'Territory', 'VI': 'Territory', 'TT': 'Territory'
}

def get_region(state: str) -> str:
    return US_REGIONS.get(state, 'Other')

def get_coast(state: str) -> str:
    """Classify as East Coast, West Coast, or Interior"""
    east_coast = ['ME', 'NH', 'MA', 'RI', 'CT', 'NY', 'NJ', 'DE', 'MD', 'VA', 'NC', 'SC', 'GA', 'FL']
    west_coast = ['WA', 'OR', 'CA', 'AK', 'HI']
    
    if state in east_coast:
        return 'East Coast'
    elif state in west_coast:
        return 'West Coast'
    else:
        return 'Interior'

def derive_origin_geography(df: DataFrame) -> DataFrame:
    df["OriginRegion"] = df["OriginState"].apply(get_region)
    df["OriginCoast"] = df["OriginState"].apply(get_coast)
    return df

print("Analyzing Origin geography granularity...")
data_ext: DataFrame = derive_origin_geography(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "OriginState", ["OriginRegion", "OriginCoast", "OriginState"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_origin.png")
plt.close()

def derive_dest_geography(df: DataFrame) -> DataFrame:
    df["DestRegion"] = df["DestState"].apply(get_region)
    df["DestCoast"] = df["DestState"].apply(get_coast)
    return df

print("Analyzing Destination geography granularity...")
data_ext: DataFrame = derive_dest_geography(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "DestState", ["DestRegion", "DestCoast", "DestState"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_destination.png")
plt.close()


# ==========================================
# ORDINAL GRANULARITY (NON-TEMPORAL)
# ==========================================

# DepDelayMinutes - has natural order: 0 (on-time) < 15-60 (moderate) < 60+ (severe)
def get_dep_delay_binary(delay: float) -> str:
    if delay <= 15:
        return "On-Time"
    else:
        return "Delayed"

def get_dep_delay_category(delay: float) -> str:
    if delay <= 0:
        return "Early/On-Time"
    elif delay <= 15:
        return "Minor Delay (1-15 min)"
    elif delay <= 60:
        return "Moderate Delay (15-60 min)"
    else:
        return "Severe Delay (60+ min)"

def get_dep_delay_detailed(delay: float) -> str:
    if delay <= 0:
        return "Early/On-Time"
    elif delay <= 15:
        return "1-15 min"
    elif delay <= 30:
        return "15-30 min"
    elif delay <= 60:
        return "30-60 min"
    elif delay <= 120:
        return "60-120 min"
    else:
        return "120+ min"

def derive_dep_delay(df: DataFrame) -> DataFrame:
    df["DepDelayBinary"] = df["DepDelayMinutes"].apply(get_dep_delay_binary)
    df["DepDelayCategory"] = df["DepDelayMinutes"].apply(get_dep_delay_category)
    df["DepDelayDetailed"] = df["DepDelayMinutes"].apply(get_dep_delay_detailed)
    return df

print("Analyzing Departure Delay granularity...")
data_ext = derive_dep_delay(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "DepDelayMinutes", ["DepDelayBinary", "DepDelayCategory", "DepDelayDetailed", "DepDelayMinutes"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_dep_delay.png")
plt.close()


# ArrDelayMinutes - has natural order: 0 (on-time) < 15-60 (moderate) < 60+ (severe)
def get_arr_delay_binary(delay: float) -> str:
    if delay <= 15:
        return "On-Time"
    else:
        return "Delayed"

def get_arr_delay_category(delay: float) -> str:
    if delay <= 0:
        return "Early/On-Time"
    elif delay <= 15:
        return "Minor Delay (1-15 min)"
    elif delay <= 60:
        return "Moderate Delay (15-60 min)"
    else:
        return "Severe Delay (60+ min)"

def get_arr_delay_detailed(delay: float) -> str:
    if delay <= 0:
        return "Early/On-Time"
    elif delay <= 15:
        return "1-15 min"
    elif delay <= 30:
        return "15-30 min"
    elif delay <= 60:
        return "30-60 min"
    elif delay <= 120:
        return "60-120 min"
    else:
        return "120+ min"

def derive_arr_delay(df: DataFrame) -> DataFrame:
    df["ArrDelayBinary"] = df["ArrDelayMinutes"].apply(get_arr_delay_binary)
    df["ArrDelayCategory"] = df["ArrDelayMinutes"].apply(get_arr_delay_category)
    df["ArrDelayDetailed"] = df["ArrDelayMinutes"].apply(get_arr_delay_detailed)
    return df

print("Analyzing Arrival Delay granularity...")
data_ext = derive_arr_delay(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "ArrDelayMinutes", ["ArrDelayBinary", "ArrDelayCategory", "ArrDelayDetailed", "ArrDelayMinutes"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_arr_delay.png")
plt.close()


# Distance - has natural order: short < medium < long haul
def get_distance_binary(distance: float) -> str:
    if distance <= 500:
        return "Short Haul"
    else:
        return "Long Haul"

def get_distance_category(distance: float) -> str:
    if distance <= 500:
        return "Short Haul (<500 mi)"
    elif distance <= 1500:
        return "Medium Haul (500-1500 mi)"
    else:
        return "Long Haul (1500+ mi)"

def get_distance_detailed(distance: float) -> str:
    if distance <= 250:
        return "Very Short (<250 mi)"
    elif distance <= 500:
        return "Short (250-500 mi)"
    elif distance <= 1000:
        return "Medium (500-1000 mi)"
    elif distance <= 1500:
        return "Medium-Long (1000-1500 mi)"
    elif distance <= 2500:
        return "Long (1500-2500 mi)"
    else:
        return "Very Long (2500+ mi)"

def derive_distance(df: DataFrame) -> DataFrame:
    df["DistanceBinary"] = df["Distance"].apply(get_distance_binary)
    df["DistanceCategory"] = df["Distance"].apply(get_distance_category)
    df["DistanceDetailed"] = df["Distance"].apply(get_distance_detailed)
    return df

print("Analyzing Distance granularity...")
data_ext = derive_distance(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "Distance", ["DistanceBinary", "DistanceCategory", "DistanceDetailed", "Distance"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_distance.png")
plt.close()


# AirTime - has natural order: short < medium < long flight duration
def get_airtime_binary(airtime: float) -> str:
    if airtime <= 120:
        return "Short Flight"
    else:
        return "Long Flight"

def get_airtime_category(airtime: float) -> str:
    if airtime <= 60:
        return "Very Short (<1h)"
    elif airtime <= 120:
        return "Short (1-2h)"
    elif airtime <= 240:
        return "Medium (2-4h)"
    else:
        return "Long (4+ h)"

def get_airtime_detailed(airtime: float) -> str:
    if airtime <= 60:
        return "<1h"
    elif airtime <= 90:
        return "1-1.5h"
    elif airtime <= 120:
        return "1.5-2h"
    elif airtime <= 180:
        return "2-3h"
    elif airtime <= 240:
        return "3-4h"
    elif airtime <= 360:
        return "4-6h"
    else:
        return "6+ h"

def derive_airtime(df: DataFrame) -> DataFrame:
    df["AirTimeBinary"] = df["AirTime"].apply(get_airtime_binary)
    df["AirTimeCategory"] = df["AirTime"].apply(get_airtime_category)
    df["AirTimeDetailed"] = df["AirTime"].apply(get_airtime_detailed)
    return df

print("Analyzing AirTime granularity...")
data_ext = derive_airtime(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "AirTime", ["AirTimeBinary", "AirTimeCategory", "AirTimeDetailed", "AirTime"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_airtime.png")
plt.close()


# ==========================================
# CATEGORICAL GRANULARITY
# ==========================================

# Airline - group by airline type/size
def get_airline_type(airline: str) -> str:
    major_carriers = ['American', 'Delta', 'United', 'Southwest', 'Alaska']
    regional = ['SkyWest', 'Endeavor', 'Republic', 'PSA', 'Envoy', 'Piedmont', 'Mesa', 'Commutair', 'GoJet', 'Air Wisconsin', 'Trans States']
    low_cost = ['Spirit', 'Frontier', 'Allegiant', 'JetBlue', 'Sun Country']
    
    airline_lower = str(airline).lower()
    
    if any(carrier.lower() in airline_lower for carrier in major_carriers):
        return "Major Carrier"
    elif any(carrier.lower() in airline_lower for carrier in regional):
        return "Regional Carrier"
    elif any(carrier.lower() in airline_lower for carrier in low_cost):
        return "Low-Cost Carrier"
    else:
        return "Other Carrier"

def get_airline_binary(airline: str) -> str:
    major_carriers = ['American', 'Delta', 'United', 'Southwest', 'Alaska']
    airline_lower = str(airline).lower()
    
    if any(carrier.lower() in airline_lower for carrier in major_carriers):
        return "Major"
    else:
        return "Regional/Other"

def derive_airline(df: DataFrame) -> DataFrame:
    df["AirlineBinary"] = df["Airline"].apply(get_airline_binary)
    df["AirlineType"] = df["Airline"].apply(get_airline_type)
    return df

print("Analyzing Airline granularity...")
data_ext = derive_airline(Combined_Flights_data.copy())
analyse_property_granularity(data_ext, "Airline", ["AirlineBinary", "AirlineType", "Airline"])
plt.tight_layout()
plt.savefig(f"{savefig_path_prefix}_granularity_airline.png")
plt.close()


print("\nGranularity analysis completed!")
print(f"All charts saved to: {os.path.dirname(savefig_path_prefix)}")