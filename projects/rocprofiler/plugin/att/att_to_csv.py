#!/usr/bin/env python3

import numpy as np
import csv
import os

def dump_csv(code, trace_instance_name, bIsAuto):
    outpath = os.getenv("OUT_FILE_NAME")
    if outpath is None:
        outpath = "att_output"
    elif os.path.dirname(outpath) != '':
        os.makedirs(os.path.dirname(outpath), exist_ok=True)

    outpath += '_' + os.path.basename(trace_instance_name) + '.csv'
    print('Generating', outpath)

    with open(outpath, 'w') as f:
        writer = csv.writer(f)
        if bIsAuto:
            writer.writerow(['Addr', 'Instruction', 'Hitcount', 'Cycles', 'C++ Reference'])
            [writer.writerow([hex(m[6]), m[0], m[7], m[8], m[3]]) for m in code]
        else:
            writer.writerow(['Line', 'Instruction', 'Hitcount', 'Cycles', 'C++ Reference'])
            [writer.writerow([m[5], m[0], m[7], m[8], m[3]]) for m in code]
