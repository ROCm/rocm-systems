import csv
import os
import re

def lit_to_signed32(match):
    """Convert a lit(0x...) value to its two's complement signed 32-bit representation."""
    hex_str = match.group(1)
    value = int(hex_str, 16)
    # Convert to signed 32-bit using two's complement
    if value >= 0x80000000:
        value -= 0x100000000
    return str(value)

def normalize_literals(text):
    """Replace lit(0x...) with the signed 32-bit two's complement value, and lit64(0x...) with just 0x..."""
    text = re.sub(r'lit\(0x([0-9a-fA-F]+)\)', lit_to_signed32, text)
    text = re.sub(r'lit64\((0x[0-9a-fA-F]+)\)', r'\1', text)
    text = text.replace('HW_REG_WAVE_SCHED_MODE', 'HW_REG_SCHED_MODE')
    return text

def normalize_row(row):
    """Apply literal normalization to each cell in a row."""
    return [normalize_literals(cell) for cell in row]

def strip_source_column(header, data):
    """Remove the 'Source' column (if present) from the header and all data rows."""
    if not header:
        return header, data
    try:
        idx = [h.strip() for h in header].index('Source')
    except ValueError:
        return header, data
    header = header[:idx] + header[idx+1:]
    data = [row[:idx] + row[idx+1:] for row in data]
    return header, data

def diff_csv_files(file1, file2):
    """
    Compare two CSV files and write the differences to an output file.
    
    :param file1: Path to the first CSV file.
    :param file2: Path to the second CSV file.
    """
    with open(file1, 'r', newline='') as f1, open(file2, 'r', newline='') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        data1 = list(reader1)
        data2 = list(reader2)

    # Normalize literals before any comparison
    data1 = [normalize_row(row) for row in data1]
    data2 = [normalize_row(row) for row in data2]

    # Strip 'Source' column — it depends on the compiler/toolchain and is not stable
    if data1:
        header1, data1[1:] = strip_source_column(data1[0], data1[1:])
        data1[0] = header1
    if data2:
        header2, data2[1:] = strip_source_column(data2[0], data2[1:])
        data2[0] = header2

    for row1, row2 in zip(data1, data2):
        if row1 != row2:
            print('Stats:', row1)
            print('Compare:', row2)

    for row in data2:
        if row not in data1:
            print('Missing:', row)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Compare two CSV files and output the differences.')
    parser.add_argument('file1', type=str, help='Path to the first CSV file.')
    parser.add_argument('file2', type=str, help='Path to the second CSV file.')

    args = parser.parse_args()

    diff_csv_files(args.file1, args.file2)
