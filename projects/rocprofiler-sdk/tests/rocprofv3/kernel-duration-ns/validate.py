#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-input",
        type=Path,
        required=True,
        help="Path to the generated kernel_trace CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path: Path = args.csv_input

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError("CSV has no header row")

        # Build a mapping from column name to index
        col_idx = {name: i for i, name in enumerate(header)}

        required_cols = ["Start_Timestamp", "End_Timestamp", "Duration_NS"]
        missing = [c for c in required_cols if c not in col_idx]
        if missing:
            raise RuntimeError(f"Missing required columns: {missing}")

        s_idx = col_idx["Start_Timestamp"]
        e_idx = col_idx["End_Timestamp"]
        d_idx = col_idx["Duration_NS"]

        # Only check the first few rows to avoid processing very large files
        checked_rows = 0
        for row in reader:
            if not row:
                continue

            start = int(row[s_idx])
            end = int(row[e_idx])
            duration = int(row[d_idx])
            calc = end - start

            if duration != calc:
                raise RuntimeError(
                    f"Duration_NS mismatch: got {duration}, expected {calc} "
                    f"(start={start}, end={end})"
                )

            checked_rows += 1
            if checked_rows >= 5:
                break

        if checked_rows == 0:
            raise RuntimeError("No data rows found to validate Duration_NS")

    print(f"[OK] Validated Duration_NS for {checked_rows} row(s) in {csv_path}")


if __name__ == "__main__":
    main()
