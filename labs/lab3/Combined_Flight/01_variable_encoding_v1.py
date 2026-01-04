import argparse
import os
import sys
from math import pi
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import cloudpickle
import unicodedata

"""

python3 01_variable_encoding_v1.py --create-joblib --joblib-path flights_2022_encoder.joblib


python3 01_variable_encoding_v1.py --csv Combined_Flights_2022.csv --out flights_2022_encoding_LE.csv

"""

class FlightEncoderV1(BaseEstimator, TransformerMixin):
    """
    Reproducible encoder for the v1 feature engineering used on the
    Combined_Flights_2022 dataset. Implements:
      - Cyclic encodings for time-related variables
      - Ordinal mapping for time blocks
      - Airline/codeshare mappings
      - Tail_Number stable integer mapping (learned on fit)
      - State risk score mapping

    Notes:
      - Unknown/unseen categorical values are mapped to -1 (or left as NaN for floats).
      - Original cyclic source columns are dropped after adding sin/cos pairs.
    """

    def __init__(self, drop_original_cyclic: bool = True):
        self.drop_original_cyclic = drop_original_cyclic

        # 2.1. ORDINAL BLOCK TIMES (string ranges -> ordinal 0..18)
        self.timeblock_order: Dict[str, int] = {
            '0001-0559': 0, '0600-0659': 1, '0700-0759': 2, '0800-0859': 3,
            '0900-0959': 4, '1000-1059': 5, '1100-1159': 6, '1200-1259': 7,
            '1300-1359': 8, '1400-1459': 9, '1500-1559': 10, '1600-1659': 11,
            '1700-1759': 12, '1800-1859': 13, '1900-1959': 14, '2000-2059': 15,
            '2100-2159': 16, '2200-2259': 17, '2300-2359': 18
        }

        # 3. Airlines (fixed mappings as in the original script)

        # Ordered based on market cap
        self.marketing_order: Dict[str, int] = {
            'NK': 0,  # Spirit
            'HA': 1,  # Hawaiian
            'F9': 2,  # Frontier
            'G4': 3,  # Allegiant
            'B6': 4,  # JetBlue
            'AS': 5,  # Alaska
            'AA': 6,  # American
            'WN': 7,  # Southwest
            'UA': 8,  # United
            'DL': 9   # Delta
        }


        # Ordered based on market cap
        self.operating_order: Dict[str, int] = {
            'G7': 8, 'C5': 0, 'ZW': 1, 'PT': 3, 'QX': 11, 'YV': 7,
            'HA': 12, 'G4': 9, 'F9': 14, 'NK': 13, 'AS': 15,
            'OH': 4, '9E': 2, 'MQ': 6, 'B6': 16, 'YX': 5,
            'OO': 10, 'UA': 19, 'AA': 18, 'DL': 20, 'WN': 17
        }
        self.codeshare_order: Dict[str, int] = {
            'HA': 12, 'G4': 9, 'F9': 14, 'NK': 13, 'AS_CODESHARE': 15, 'AS': 15,
            'B6': 16, 'DL_CODESHARE': 20, 'DL': 20, 'UA_CODESHARE': 19, 'UA': 19,
            'AA_CODESHARE': 18, 'AA': 18, 'WN': 17
        }

        # 5. States (FEMA-inspired ranking used in prior version)
        self.state_fips_order: Dict[str, float] = {
            'TT': 0, 'PR': 1, 'VI': 2, 'HI': 30.36, 'AK': 25.00, 'CA': 100.00,
            'OR': 78.57, 'WA': 89.29, 'NV': 57.14, 'AZ': 37.50, 'UT': 64.29,
            'ID': 26.79, 'MT': 23.21, 'WY': 14.29, 'CO': 53.57, 'NM': 28.57,
            'ND': 32.14, 'SD': 35.71, 'NE': 42.86, 'KS': 46.43, 'OK': 66.07,
            'TX': 96.43, 'MN': 51.79, 'IA': 62.50, 'MO': 80.36, 'AR': 58.93,
            'LA': 92.86, 'WI': 41.07, 'IL': 85.71, 'MS': 71.43, 'MI': 60.71,
            'IN': 50.00, 'KY': 55.36, 'TN': 73.21, 'AL': 75.00, 'OH': 48.21,
            'WV': 19.64, 'FL': 98.21, 'GA': 82.14, 'SC': 87.50, 'NC': 91.07,
            'VA': 67.86, 'MD': 44.64, 'DE': 16.07, 'PA': 69.64, 'NJ': 83.93,
            'NY': 76.79, 'CT': 33.93, 'RI': 12.50, 'MA': 39.29, 'VT': 5.36,
            'NH': 17.86, 'ME': 21.43
        }

        # Cyclic periods and which are 1-based domains
        self.cyclic_periods: Dict[str, int] = {
            'Quarter': 4,
            'Month': 12,
            'DayOfWeek': 7,
            'DayofMonth': 31,
            'DepTimeBlk': 19,
            'ArrTimeBlk': 19,
        }
        self._one_based: set = {'Quarter', 'Month', 'DayOfWeek', 'DayofMonth'}

        # Learned on fit
        self.tail_map_: Optional[Dict[str, int]] = None
        self.origin_airport_map_: Optional[Dict[str, int]] = None
        self.origin_city_market_map_: Optional[Dict[str, int]] = None

    def _map_tail_numbers(self, s: pd.Series) -> pd.Series:
        if self.tail_map_ is None:
            # No fitting done; return NaNs
            return pd.Series([-1] * len(s), index=s.index, dtype='int64')
        vals = s.astype('string').fillna('<NA>')
        return vals.map(self.tail_map_).fillna(-1).astype('int64')

    def _map_origin_airport_id(self, s: pd.Series) -> pd.Series:
        if self.origin_airport_map_ is None:
            return pd.Series([-1] * len(s), index=s.index, dtype='int64')
        vals = s.astype('string').fillna('<NA>')
        return vals.map(self.origin_airport_map_).fillna(-1).astype('int64')

    def _map_origin_city_market_id(self, s: pd.Series) -> pd.Series:
        if self.origin_city_market_map_ is None:
            return pd.Series([-1] * len(s), index=s.index, dtype='int64')
        vals = s.astype('string').fillna('<NA>')
        return vals.map(self.origin_city_market_map_).fillna(-1).astype('int64')

    @staticmethod
    def _numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='coerce')

    @staticmethod
    def _encode_flight_date(series: pd.Series) -> pd.Series:
        """Convert FlightDate into a numeric value.

        Accepts strings like "01-02-2023" or "01/02/2023" and assumes
        day-first format (DD-MM-YYYY or DD/MM/YYYY). The output is the
        number of days since the Unix epoch (1970-01-01) as float; invalid
        or missing dates become NaN.
        """
        # If it's already datetime-like, keep as is; otherwise parse.
        if not np.issubdtype(series.dtype, np.datetime64):
            # Unify separators so pandas doesn't get confused by mixed usage.
            s = series.astype("string").str.replace("/", "-", regex=False)
            dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
        else:
            dt = series

        # Map NaT to NaN and valid dates to integer day offsets from epoch.
        result = pd.Series(np.nan, index=series.index, dtype="float64")
        mask = dt.notna()
        if mask.any():
            # Use days since epoch (1970-01-01) for a compact numeric scale.
            days = (dt[mask].view("int64") // 86_400_000_000_000)
            result.loc[mask] = days.astype("float64")
        return result

    @staticmethod
    def _normalize_token(val) -> str:
        """Normalize a value to a lowercase, accent-free token for comparison."""
        if isinstance(val, (bool, np.bool_)):
            return 'true' if bool(val) else 'false'
        try:
            s = str(val)
        except Exception:
            return ''
        s = s.strip()
        # Remove accents/diacritics and lower-case
        s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
        return s.lower()

    @classmethod
    def _to_binary(cls, val) -> int:
        """
        Convert arbitrary input to binary 0/1 with broad language support.
        Truthy tokens: TRUE/true/Verdadeiro/verdadeiro/sim/SIM/Sim and numeric non-zero.
        Falsy tokens: FALSE/false/Falso/falso/nao/não/no and numeric zero.
        Missing/unknown values default to 0.
        """
        # Missing -> 0
        if pd.isna(val):
            return 0
        # Booleans directly
        if isinstance(val, (bool, np.bool_)):
            return 1 if bool(val) else 0
        # Numerics: non-zero -> 1, zero -> 0
        if isinstance(val, (int, np.integer)):
            return 1 if int(val) != 0 else 0
        if isinstance(val, (float, np.floating)):
            try:
                return 1 if int(float(val)) != 0 else 0
            except Exception:
                pass
        # Strings/tokens
        tok = cls._normalize_token(val)
        if tok == '':
            return 0
        truthy = {'true', 'verdadeiro', 'sim', 'yes', 'y', '1'}
        falsy = {'false', 'falso', 'nao', 'no', 'n', '0'}
        if tok in truthy:
            return 1
        if tok in falsy:
            return 0
        # Try numeric string fallback
        try:
            return 1 if int(tok) != 0 else 0
        except Exception:
            return 0

    @classmethod
    def _map_cancelled(cls, s: pd.Series) -> pd.Series:
        """Map Cancelled column to strictly binary 0/1 per specification."""
        return s.apply(cls._to_binary).astype('int64')

    def _encode_cyclic(self, df: pd.DataFrame, col: str, period: int) -> None:
        if col not in df.columns:
            return
        s = self._numeric(df[col])
        if col in self._one_based:
            s = s - 1  # convert 1..period -> 0..period-1
        s = s % period
        angle = 2 * pi * (s / period)
        df[f"{col}_sin"] = np.sin(angle)
        df[f"{col}_cos"] = np.cos(angle)
        if self.drop_original_cyclic:
            df.drop(columns=[col], inplace=True)

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if 'Tail_Number' in df.columns:
            cats = (
                df['Tail_Number']
                .dropna()
                .astype('string')
                .unique()
                .tolist()
            )
            # Stable, deterministic order
            cats = sorted(cats)
            self.tail_map_ = {c: i for i, c in enumerate(cats)}
        else:
            self.tail_map_ = {}

        # Build a unified AirportID map from both Origin and Dest
        airport_ids: List[str] = []
        if 'OriginAirportID' in df.columns:
            airport_ids.extend(
                df['OriginAirportID']
                .dropna()
                .astype('string')
                .unique()
                .tolist()
            )
        if 'DestAirportID' in df.columns:
            airport_ids.extend(
                df['DestAirportID']
                .dropna()
                .astype('string')
                .unique()
                .tolist()
            )
        airport_ids = sorted(set(airport_ids))
        self.origin_airport_map_ = {c: i for i, c in enumerate(airport_ids)}

        # Build a unified CityMarketID map from both Origin and Dest
        city_ids: List[str] = []
        if 'OriginCityMarketID' in df.columns:
            city_ids.extend(
                df['OriginCityMarketID']
                .dropna()
                .astype('string')
                .unique()
                .tolist()
            )
        if 'DestCityMarketID' in df.columns:
            city_ids.extend(
                df['DestCityMarketID']
                .dropna()
                .astype('string')
                .unique()
                .tolist()
            )
        city_ids = sorted(set(city_ids))
        self.origin_city_market_map_ = {c: i for i, c in enumerate(city_ids)}

        
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # 2.1. Time blocks -> ordinal (if strings)
        for col in ['DepTimeBlk', 'ArrTimeBlk']:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].map(self.timeblock_order)

        # 3. Airline-related categorical mappings
        if 'Marketing_Airline_Network' in df.columns:
            df['Marketing_Airline_Network'] = df['Marketing_Airline_Network'].map(self.marketing_order)
        if 'Operating_Airline' in df.columns:
            df['Operating_Airline'] = df['Operating_Airline'].map(self.operating_order)
        if 'Operated_or_Branded_Code_Share_Partners' in df.columns:
            df['Operated_or_Branded_Code_Share_Partners'] = df['Operated_or_Branded_Code_Share_Partners'].map(self.codeshare_order)

        # 4. Tail number stable integer codes
        if 'Tail_Number' in df.columns:
            df['Tail_Number'] = self._map_tail_numbers(df['Tail_Number'])

        # 5. States mapping
        if 'OriginState' in df.columns:
            df['OriginState'] = df['OriginState'].map(self.state_fips_order)
        if 'DestState' in df.columns:
            df['DestState'] = df['DestState'].map(self.state_fips_order)

        # OriginAirportID -> learned label encoding 0..N-1
        if 'OriginAirportID' in df.columns:
            df['OriginAirportID'] = self._map_origin_airport_id(df['OriginAirportID'])

        # DestAirportID -> use the same learned AirportID map
        if 'DestAirportID' in df.columns:
            df['DestAirportID'] = self._map_origin_airport_id(df['DestAirportID'])

        # Origin/Dest CityMarketID -> learned label encoding 0..N-1 (unified map)
        if 'OriginCityMarketID' in df.columns:
            df['OriginCityMarketID'] = self._map_origin_city_market_id(df['OriginCityMarketID'])
        if 'DestCityMarketID' in df.columns:
            df['DestCityMarketID'] = self._map_origin_city_market_id(df['DestCityMarketID'])

        # 6. FlightDate -> numeric (days since epoch)
        if 'FlightDate' in df.columns:
            df['FlightDate'] = self._encode_flight_date(df['FlightDate'])

        # Cancelled -> enforce binary mapping (TRUE/true/Verdadeiro/sim -> 1; else -> 0)
        if 'Cancelled' in df.columns:
            df['Cancelled'] = self._map_cancelled(df['Cancelled'])

        # 2. Cyclic variables (including time blocks after ordinalization)
        for col, period in self.cyclic_periods.items():
            self._encode_cyclic(df, col, period)

        return df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the encoding pipeline on a CSV, or create a joblib encoder."
        )
    )
    parser.add_argument(
        "--csv",
        default="Combined_Flights_2022.csv",
        help="Path to the CSV dataset (default: Combined_Flights_2022.csv)",
    )
    parser.add_argument(
        "--out",
        default="flights_2022_encoding_LE.csv",
        help="Output CSV path when applying the encoder (default: flights_2022_encoding_LE.csv)",
    )
    parser.add_argument(
        "--create-joblib",
        action="store_true",
        help="If set, do not output CSV; create a joblib encoder instead.",
    )
    parser.add_argument(
        "--joblib-path",
        default="flights_2022_encoder.joblib",
        help="Output path for the joblib encoder (default: flights_2022_encoder.joblib)",
    )
    return parser.parse_args(argv)


def run_encoding(csv_path: str, out_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {csv_path} — shape={df.shape}")

    encoder = FlightEncoderV1(drop_original_cyclic=True)
    out_df = encoder.fit_transform(df)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Saved encoded dataset to: {out_path} — new shape={out_df.shape}")


def create_encoder_joblib(csv_path: str, joblib_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset for fitting encoder: {csv_path} — shape={df.shape}")

    encoder = FlightEncoderV1(drop_original_cyclic=True)
    encoder.fit(df)
    # Use cloudpickle to serialize custom class objects to avoid module path issues
    # when loading in different execution contexts (e.g., uvicorn workers).
    with open(joblib_path, 'wb') as f:
        cloudpickle.dump(encoder, f)
    print(f"Created cloudpickle-serialized encoder at: {joblib_path}")


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.create_joblib:
            create_encoder_joblib(args.csv, args.joblib_path)
        else:
            run_encoding(args.csv, args.out)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())