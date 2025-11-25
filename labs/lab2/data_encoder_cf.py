#!/usr/bin/env python3
"""Encode Combined_Flights_2022.csv similarly to traffic_accidents encoder.

Variable groups (design choices; adjust if analysis suggests refinements):
NUMERIC_WITH_CONVERSION: FlightDate (timestamp string -> epoch seconds)
CYCLICAL (time-of-day in HHMM local): CRSDepTime, DepTime, CRSArrTime, ArrTime, WheelsOff, WheelsOn
ORDINAL (boolean / ordered categories): Cancelled, Diverted, DepDel15, ArrDel15,
	DepartureDelayGroups, ArrivalDelayGroups, DistanceGroup, DivAirportLandings,
	(time-block ranges treated as ordered): DepTimeBlk, ArrTimeBlk
NOMINAL (one-hot): Airline, Origin, Dest, Marketing_Airline_Network,
	Operated_or_Branded_Code_Share_Partners, IATA_Code_Marketing_Airline,
	Operating_Airline, IATA_Code_Operating_Airline, Tail_Number,
	OriginCityName, OriginState, OriginStateName, DestCityName, DestState,
	DestStateName
NUMERIC_NO_CONVERSION: All remaining already-numeric columns (IDs, counts, delays, durations).

Notes:
 - Time HHMM values converted to minutes since midnight then encoded via sin/cos; originals dropped.
 - Block columns DepTimeBlk/ArrTimeBlk mapped in chronological order of their starting HHMM (e.g. "0500-0559" < "0600-0659").
 - Booleans/True/False mapped to 0/1.
 - Script is defensive: skips columns not present; dynamic mapping printed if unexpected block formats appear.
"""

from __future__ import annotations

from pathlib import Path
from math import pi, sin, cos
import pandas as pd

INPUT_PATH = Path(__file__).parent / ".." / ".." / "classification" / "Combined_Flights_2022.csv"
OUTPUT_PATH = Path(__file__).parent / ".." / ".." / "classification" / "Combined_Flights_2022_encoded.csv"

# Column groups
CYCLICAL = ["CRSDepTime", "DepTime", "CRSArrTime", "ArrTime", "WheelsOff", "WheelsOn"]
ORDINAL = [
	"Cancelled", "Diverted", "DepDel15", "ArrDel15",
	"DepartureDelayGroups", "ArrivalDelayGroups", "DistanceGroup", "DivAirportLandings",
	"DepTimeBlk", "ArrTimeBlk",
]
NOMINAL = [
	"Airline", "Origin", "Dest", "Marketing_Airline_Network",
	"Operated_or_Branded_Code_Share_Partners", "IATA_Code_Marketing_Airline",
	"Operating_Airline", "IATA_Code_Operating_Airline", "Tail_Number",
	"OriginCityName", "OriginState", "OriginStateName", "DestCityName",
	"DestState", "DestStateName",
]
NUMERIC_WITH_CONVERSION = ["FlightDate"]

# Basic boolean/ordinal mappings (others dynamically generated)
ordinal_mappings: dict[str, dict] = {
	"Cancelled": {False: 0, True: 1, "False": 0, "True": 1},
	"Diverted": {False: 0, True: 1, "False": 0, "True": 1},
	"DepDel15": {0: 0, 1: 1, 0.0: 0, 1.0: 1},
	"ArrDel15": {0: 0, 1: 1, 0.0: 0, 1.0: 1},
}


def generate_mapping(values: list) -> dict:
	"""Alphabetical mapping for categorical values."""
	return {v: i for i, v in enumerate(sorted(values, key=lambda x: str(x)))}


def parse_block_start(block: str) -> int:
	"""Extract numeric HHMM start from a time block string like '0500-0559'."""
	if not isinstance(block, str) or "-" not in block:
		return 9999  # push unknowns to end
	start = block.split("-")[0]
	# Remove any whitespace
	start = start.strip()
	return int(start) if start.isdigit() else 9999


def encode_time_cyclical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
	for c in cols:
		if c not in df.columns:
			continue
		def to_minutes(val):
			if pd.isna(val):
				return None
			try:
				iv = int(float(val))
				hour = iv // 100
				minute = iv % 100
				if hour > 23 or minute > 59:
					return None
				return hour * 60 + minute
			except Exception:
				return None
		minutes = df[c].apply(to_minutes)
		max_minutes = 1440
		df[c + "_sin"] = minutes.apply(lambda m: 0.0 if m is None else round(sin(2 * pi * m / max_minutes), 6))
		df[c + "_cos"] = minutes.apply(lambda m: 1.0 if m is None else round(cos(2 * pi * m / max_minutes), 6))
	existing = [c for c in cols if c in df.columns]
	return df.drop(columns=existing)


def main():
	df = pd.read_csv(INPUT_PATH)

	# FlightDate -> epoch seconds
	if "FlightDate" in df.columns:
		dt = pd.to_datetime(df["FlightDate"], errors="coerce")
		df["FlightDate"] = (dt - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

	# Ordinal encoding
	for col in ORDINAL:
		if col not in df.columns:
			continue
		uniques = [u for u in df[col].dropna().unique()]
		mapping = ordinal_mappings.get(col, {})
		# For block columns, order by start time
		if col in ("DepTimeBlk", "ArrTimeBlk"):
			try:
				ordered = sorted(uniques, key=parse_block_start)
				mapping = {v: i for i, v in enumerate(ordered)}
				ordinal_mappings[col] = mapping
			except Exception:
				mapping = generate_mapping(uniques)
				ordinal_mappings[col] = mapping
		elif not mapping:
			# If values numeric already, leave as-is
			if all(isinstance(u, (int, float)) for u in uniques):
				continue
			mapping = generate_mapping(uniques)
			ordinal_mappings[col] = mapping
		# Validate coverage
		if set(uniques) - set(mapping.keys()):
			missing = set(uniques) - set(mapping.keys())
			auto_map = generate_mapping(uniques)
			print(f"Warning: Missing keys for {col}: {missing}. Auto mapping used: {auto_map}")
			mapping = auto_map
			ordinal_mappings[col] = mapping
		df[col] = df[col].replace(mapping)

	# One-hot nominal
	nominal_existing = [c for c in NOMINAL if c in df.columns]
	if nominal_existing:
		dummies = pd.get_dummies(df[nominal_existing], prefix=nominal_existing, dummy_na=False)
		df = pd.concat([df.drop(columns=nominal_existing), dummies], axis=1)

	# Cyclical time encoding
	df = encode_time_cyclical(df, CYCLICAL)

	df.to_csv(OUTPUT_PATH, index=False)
	print(f"Encoded flights dataset written to: {OUTPUT_PATH}")
	# Optional: show mappings applied
	for k, v in ordinal_mappings.items():
		if v:
			print(f"Mapping {k}: {v}")


if __name__ == "__main__":
	main()

