#!/usr/bin/env python3

import numpy as np
import csv
import os

def dump_csv(code, trace_instance_name):
    outpath = os.getenv("OUT_FILE_NAME")
    if outpath is None:
        outpath = "att_output"
    outpath += '_' + trace_instance_name.split('/')[-1] + '.csv'

    with open(outpath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Line', 'Instruction', 'Hitcount', 'Cycles', 'Addr', 'C++ Reference'])
        [writer.writerow([m[5], m[0], m[7], m[8], hex(m[6]), m[3]]) for m in code]
