import argparse
import os
import sys
from typing import List

"""
Varibles:

FlightDate - done
Cancelled - done
CRSDepTime - done
CRSElapsedTime - done
Distance - done
Quarter - done
Month - done
DayofMonth - done
DayOfWeek - done
Marketing_Airline_Network - done
Operated_or_Branded_Code_Share_Partners - done
Operating_Airline - done
Tail_Number - done
OriginAirportID - done
OriginCityMarketID - done
OriginState - done
DestAirportID - done
DestCityMarketID - done
DestState - done
DepTimeBlk - done
CRSArrTime - done
ArrTimeBlk - done
DistanceGroup - done

"""

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import cloudpickle

# python 00_variable_dropping.py --csv Combined_Flights_2022.csv
# OR to create a joblib dropper:
# python3 00_variable_dropping.py --create-joblib --joblib-path flights_2022_dropper.joblib

# Columns to drop (leakage/redundant/unused)
TO_DROP: List[str] = [
	"Flight_Number_Operating_Airline",
	"DOT_ID_Marketing_Airline",
	"Flight_Number_Marketing_Airline",
	"Diverted",
	"DivAirportLandings",
	"ArrivalDelayGroups",
	"ArrDel15",
	"ArrDelay",
	"TaxiIn",
	"TaxiOut",
	"WheelsOn",
	"WheelsOff",
	"AirTime",
	"DepartureDelayGroups",
	"DepDel15",
	"ActualElapsedTime",
	"ArrDelayMinutes",
	"ArrTime",
	"DepDelay",
	"DepDelayMinutes",
	"DepTime",
	"Year",
	"OriginStateName",
	"OriginStateFips",
	"OriginWac",
	"OriginCityName",
	"OriginAirportSeqID",
	"DestAirportSeqID",
	"IATA_Code_Operating_Airline",
	"IATA_Code_Marketing_Airline",
	"DOT_ID_Operating_Airline",
	"DestStateName",
	"DestStateFips",
	"DestWac",
	"DestCityName",
	"Airline",
	"Origin",
	"Dest"
]


class ColumnDropper(BaseEstimator, TransformerMixin):
	def __init__(self, columns: List[str]):
		self.columns = list(columns)

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		df = X.copy()
		present = [c for c in self.columns if c in df.columns]
		missing = [c for c in self.columns if c not in df.columns]

		for c in present:
			print(f"[dropper] Drop: {c} (exists)")
		for c in missing:
			print(f"[dropper] Skip: {c} (not found)")

		return df.drop(columns=present)


def drop_inplace(csv_path: str) -> None:
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	print(f"Loaded dataset: {csv_path} — shape={df.shape}")

	present = [c for c in TO_DROP if c in df.columns]
	missing = [c for c in TO_DROP if c not in df.columns]

	for c in present:
		print(f"Drop: {c} — exists")
	for c in missing:
		print(f"Skip: {c} — not found")

	new_df = df.drop(columns=present)
	new_df.to_csv(csv_path, index=False)

	print(
		f"Saved updated dataset to the same path. Old shape={df.shape}, New shape={new_df.shape}"
	)


def create_dropper_joblib(joblib_path: str) -> None:
	dropper = ColumnDropper(TO_DROP)
	# Use cloudpickle to avoid '__mp_main__' issues when loading under uvicorn.
	with open(joblib_path, 'wb') as f:
		cloudpickle.dump(dropper, f)
	print(f"Created cloudpickle-serialized dropper at: {joblib_path}")


def parse_args(argv=None):
	parser = argparse.ArgumentParser(
		description=(
			"Drop configured columns from a CSV in-place, or create a joblib dropper."
		)
	)
	parser.add_argument(
		"--csv",
		default="Combined_Flights_2022.csv",
		help="Path to the CSV dataset (default: Combined_Flights_2022.csv)",
	)
	parser.add_argument(
		"--create-joblib",
		action="store_true",
		help="If set, do not modify the CSV; create a joblib dropper instead.",
	)
	parser.add_argument(
		"--joblib-path",
		default="flights_2022_dropper.joblib",
		help="Output path for the joblib dropper (default: flights_2022_dropper.joblib)",
	)
	return parser.parse_args(argv)

def main(argv=None) -> int:
	args = parse_args(argv)

	try:
		if args.create_joblib:
			create_dropper_joblib(args.joblib_path)
		else:
			drop_inplace(args.csv)
	except Exception as e:
		print(f"Error: {e}")
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())

