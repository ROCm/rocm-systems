#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices,
# Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import argparse
import csv
import logging
from pathlib import Path


def load_csv_data(csv_path: Path):
    """
    Load CSV data and validate it.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of dictionaries containing CSV data
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        RuntimeError: If CSV is invalid or empty
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        
        # Check if header exists
        if reader.fieldnames is None:
            raise RuntimeError("CSV has no header row")
        
        # Validate required columns
        required_cols = ["Start_Timestamp", "End_Timestamp", "Duration_NS"]
        missing = [c for c in required_cols if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")
        
        # Read all rows
        for row in reader:
            rows.append(row)
    
    if not rows:
        raise RuntimeError("No data rows found in CSV")
    
    return rows


def validate_duration(csv_data):
    """
    Validate that Duration_NS = End_Timestamp - Start_Timestamp for all rows.
    
    Args:
        csv_data: List of dictionaries containing CSV data
        
    Raises:
        RuntimeError: If duration calculation is incorrect
    """
    for i, row in enumerate(csv_data):
        start = int(row["Start_Timestamp"])
        end = int(row["End_Timestamp"])
        duration = int(row["Duration_NS"])
        calc = end - start
        
        if duration != calc:
            raise RuntimeError(
                f"Row {i}: Duration_NS mismatch: got {duration}, expected {calc} "
                f"(start={start}, end={end})"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-input",
        type=Path,
        required=True,
        help="Path to the generated kernel_trace CSV",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(levelname)s: %(message)s'
    )
    
    csv_path: Path = args.csv_input
    
    # Load and validate CSV data
    csv_data = load_csv_data(csv_path)
    
    # Validate duration calculations
    validate_duration(csv_data)
    
    logging.info(f"[OK] Validated Duration_NS for {len(csv_data)} row(s) in {csv_path}")


if __name__ == "__main__":
    main()
