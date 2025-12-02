# =====================================================
# FLIGHT STATUS PREDICTION - ENCODING (Fase 1)
# =======  VERSÃO 1 (Label Encoding Encoding) ===

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from math import pi
from dslabs_functions import encode_cyclic_variables


# ------------------ CARREGAR DADOS ------------------
data = pd.read_csv("Combined_Flights_2022.csv")

print(f"Dataset carregado: {data.shape}")


# 2. CYCLIC VARIABLES


def prepare_and_encode(data: pd.DataFrame, vars: dict):
    original_len = len(data)

    fake_row = {v: vmax for v, vmax in vars.items()}
    data = pd.concat([data, pd.DataFrame([fake_row])], ignore_index=True)

    encode_cyclic_variables(data, list(vars.keys()))
    data = data.iloc[:original_len]
    return data


vars_expected = {
    "Quarter": 4,
    "Month": 12,
    "DayOfWeek": 7,
    "DayofMonth": 31}

data = prepare_and_encode(data, vars_expected)


# 2.1. ORDINAL BLOCK TIMES
timeblock_order = {
    '0001-0559': 0, '0600-0659': 1, '0700-0759': 2, '0800-0859': 3,
    '0900-0959': 4, '1000-1059': 5, '1100-1159': 6, '1200-1259': 7,
    '1300-1359': 8, '1400-1459': 9, '1500-1559': 10, '1600-1659': 11,
    '1700-1759': 12, '1800-1859': 13, '1900-1959': 14, '2000-2059': 15,
    '2100-2159': 16, '2200-2259': 17, '2300-2359': 18
}
data['DepTimeBlk'] = data['DepTimeBlk'].map(timeblock_order)
data['ArrTimeBlk'] = data['ArrTimeBlk'].map(timeblock_order)
data = prepare_and_encode(data, {'DepTimeBlk':19,'ArrTimeBlk':19})


# 3. AIRLINE  ------------
marketing_order = {
    'HA': 0, 'G4': 1, 'F9': 2, 'NK': 3, 'B6': 4,
    'AS': 5, 'WN': 6, 'DL': 7, 'UA': 8, 'AA': 9}

data['Marketing_Airline_Network'] = data['Marketing_Airline_Network'].map(marketing_order)
data['IATA_Code_Marketing_Airline'] = data['IATA_Code_Marketing_Airline'].map(marketing_order)

operating_order = {'G7': 0, 'C5': 1, 'ZW': 2, 'PT': 3, 'QX': 4, 'YV': 5,
                  'HA': 6, 'G4': 7, 'F9': 8, 'NK': 9, 'AS': 10,
                  'OH': 11, '9E': 12, 'MQ': 13, 'B6': 14, 'YX': 15,
                  'OO': 16, 'UA': 17, 'AA': 18, 'DL': 19, 'WN': 20}

data['Operating_Airline'] = data['Operating_Airline'].map(operating_order)
data['IATA_Code_Operating_Airline'] = data['IATA_Code_Operating_Airline'].map(operating_order)

data['Operated_or_Branded_Code_Share_Partners'] = data['Operated_or_Branded_Code_Share_Partners'].map({
    'HA':6, 'G4':7, 'F9':8, 'NK':9, 'AS_CODESHARE':10, 'AS':10,
    'B6':14, 'DL_CODESHARE':19, 'DL':19, 'UA_CODESHARE':17, 'UA':17,
    'AA_CODESHARE':18, 'AA':18, 'WN':20})

le_airline = LabelEncoder()
data['Airline'] = le_airline.fit_transform(data['Airline'].astype(str))


# 4. TAIL NUMBER

data['Tail_Number'] = data['Tail_Number'].astype('category').cat.codes

# 5. STATES, CITIES, AIRPORTS
# Ordem aproximada por FIPS (Sul -> Oeste ->  Nordeste ->  Centro)
state_fips_order = {
    'TT':0, 'PR':1, 'VI':2, 'HI':3, 'AK':4, 'CA':5, 'OR':6, 'WA':7,
    'NV':8, 'AZ':9, 'UT':10, 'ID':11, 'MT':12, 'WY':13, 'CO':14, 'NM':15,
    'ND':16, 'SD':17, 'NE':18, 'KS':19, 'OK':20, 'TX':21, 'MN':22, 'IA':23,
    'MO':24, 'AR':25, 'LA':26, 'WI':27, 'IL':28, 'MS':29, 'MI':30, 'IN':31,
    'KY':32, 'TN':33, 'AL':34, 'OH':35, 'WV':36, 'FL':37, 'GA':38, 'SC':39,
    'NC':40, 'VA':41, 'MD':42, 'DE':43, 'PA':44, 'NJ':45, 'NY':46, 'CT':47,
    'RI':48, 'MA':49, 'VT':50, 'NH':51, 'ME':52
}
data['OriginState'] = data['OriginState'].map(state_fips_order)
data['DestState'] = data['DestState'].map(state_fips_order)
state_name_order = {
    'U.S. Pacific Trust Territories and Possessions': 0,'Puerto Rico': 1,
    'U.S. Virgin Islands': 2,'Hawaii': 3,'Alaska': 4,'California': 5,'Oregon': 6,'Washington': 7,
    'Nevada': 8,'Arizona': 9,'Utah': 10,'Idaho': 11,'Montana': 12,'Wyoming': 13,'Colorado': 14,
    'New Mexico': 15,'North Dakota': 16,'South Dakota': 17,'Nebraska': 18,
    'Kansas': 19,'Oklahoma': 20,'Texas': 21,'Minnesota': 22,'Iowa': 23,
    'Missouri': 24, 'Arkansas': 25, 'Louisiana': 26, 'Wisconsin': 27,'Illinois': 28,
    'Mississippi': 29,'Michigan': 30,'Indiana': 31,'Kentucky': 32, 'Tennessee': 33, 'Alabama': 34,
    'Ohio': 35, 'West Virginia': 36, 'Florida': 37, 'Georgia': 38,
    'South Carolina': 39, 'North Carolina': 40, 'Virginia': 41,
    'Maryland': 42, 'Delaware': 43, 'Pennsylvania': 44,'New Jersey': 45,'New York': 46,
    'Connecticut': 47,
    'Rhode Island': 48,'Massachusetts': 49, 'Vermont': 50,'New Hampshire': 51,'Maine': 52}

data['OriginStateName'] = data['OriginStateName'].map(state_name_order)
data['DestStateName']   = data['DestStateName'].map(state_name_order)

# ----- Cities -----
le_city = LabelEncoder()
all_cities = pd.concat([data['OriginCityName'], data['DestCityName']])
le_city.fit(all_cities)

data['OriginCityName'] = le_city.transform(data['OriginCityName'])
data['DestCityName'] = le_city.transform(data['DestCityName'])


# ----- Airports -----

le_origin = LabelEncoder()
data['Origin'] = le_origin.fit_transform(data['Origin'])

le_dest = LabelEncoder()
data['Dest'] = le_dest.fit_transform(data['Dest'])
data.drop(columns=['FlightDate'], inplace=True)

data['Cancelled'] = data['Cancelled'].fillna(0).astype(int)
data['Diverted'] = data['Diverted'].fillna(0).astype(int)
#  RESULTADO

print(f"Dataset final: {data.shape}")
print("Colunas após encoding:")
print(data.columns.tolist())

data.to_csv("flights_2022_encoding_LE.csv", index=False)
print("Criado ficheiro: flights_2022_prepared_phase1.csv")