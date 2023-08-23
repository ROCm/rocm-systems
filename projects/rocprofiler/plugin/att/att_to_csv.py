#!/usr/bin/env python3

import numpy as np
import csv
import os

def dump_csv(code):
    outpath = os.getenv("OUT_FILE_NAME")
    if outpath is None:
        outpath = "att_output.csv"
    if ".csv" not in outpath:
        outpath += ".csv"

    with open(outpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Line', 'Instruction', 'Hitcount', 'Cycles', 'Addr', 'C++ Reference'])
        [writer.writerow([m[5], m[0], m[7], m[8], hex(m[6]), m[3]]) for m in code]
