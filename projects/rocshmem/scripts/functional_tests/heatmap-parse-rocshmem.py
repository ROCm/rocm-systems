import os
import sys
import re
#from matplotlib import pyplot as plt
import xlsxwriter
from datetime import datetime
from dataclasses import dataclass
from typing import List

## EDIT THESE VALUES
bkc_version = "BKC 00.25.11.02 (38.4 GT/s XGMI)"
ifwi_version = "IFWI 00.940.956978"
amdgpu_version = "amdgpu 6.14.14-2193512"
os_kernel = "CentOS Kernel 6.9.0-0_fbk10_brcmrdma13_141_g9b20106afb70"
nic_driver = "Broadcom driver=6.9.0-0_fbk10_brcmrdma13_141_g9 firmware=232.0.213.0/pkg 232.1.190.0"
###

rocshmem_version = ""
rocm_version = ""
hip_version = ""
minmsgsize = 16

if len(sys.argv) <= 1:
    print("No input directory provided. Aborting")
    sys.exit()

files_in_dir = {}
files = 0

for dir in sys.argv[1:]:
    file_names = []
    if not os.path.isdir (dir):
        continue
    for entry in os.listdir(dir):
        full_path = os.path.join(dir, entry)
        if os.path.isfile(full_path):
            file_names.append(entry)
    files_in_dir[dir] = file_names
    files = files + len(file_names)

unique="rocSHMEM_MI300_Thor2_Heatmap"

## cols => no. of consecutive runs
## rows => no. of msg sizes in the sweep -- 35 = 1B-16GB
cols, rows = 1, 27

x = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824]
x_str = [16, 32, 64, 128, 256, 512, '1KB', '2KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB']
#x = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
#x_str = [16, 32, 64, 128, 256, 512, '1KB', '2KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB']


@dataclass
class Measurement:
    volume: int
    msgsize: int
    msgcount: int
    avg_time: float
    avg_bw: float
    msg_rate: float

@dataclass
class Series:
    op: str
    ngpus: int
    nwgs: int
    nthreads: int
    data: List[Measurement]

uniq = f"rocshmem_test"
workbook = xlsxwriter.Workbook(f"{uniq}.xlsx")
workbook.set_properties({'company':  'AMD'})

cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True})
num_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': '0.00'})

now = datetime.now()
date_str = now.strftime("%Y-%m-%d");

for dir, file_names in files_in_dir.items():
    worksheet = workbook.add_worksheet(f"{dir}-{date_str}")
    worksheet.set_zoom(70)
    all_data = []

    for file in file_names:
        mpirun_cmd = ""

        filename=f"{dir}/{file}"
        #print(filename)

        elems = file.split('_')
        op = elems[0]
        procs = elems[1]
        ngpu = int(re.findall("[0-9]+",procs)[0])

        wgs = elems[2]
        nwgs = int(re.findall("[0-9]+",wgs)[0])
        threads = elems[3]
        nthreads = int(re.findall("[0-9]+",threads)[0])

        this_series = Series(op, ngpu, nwgs, nthreads, [])

        with open(f"{filename}", 'r') as file1:
            for lno1, line1 in enumerate(file1, start=0):
                if "mpirun " in line1.rstrip():
                    mpirun_cmd = line1.rstrip()
                    #print(mpirun_cmd)
                    continue

                if "#" in line1.rstrip():
                    continue

                if "[" in line1.rstrip():
                    continue

                values = line1.rstrip().split()
                #print(values)

                volume1 = int(values[0])
                size1 = int(values[1])
                msgcount1 = int(values[2])
                avg_time1 = float(values[3])
                avg_bw1 = float(values[4])
                msg_rate1 = float(values[5])

                if volume1 < minmsgsize:
                    continue

                datapoint = Measurement(volume1, size1, msgcount1, avg_time1, avg_bw1, msg_rate1)
                this_series.data.append(datapoint)
        all_data.append(this_series)

    # sort all_data such that data of the same operation appear together
    # and num_gpus for the same operation are increasing
    all_data.sort(key = lambda item: (item.op, item.ngpus))

    #for data_series in all_data:
    #    print(data_series.op, data_series.ngpus)
    #    for dt in data_series.data:
    #        print("   ", dt.msgsize, dt.avg_time)

    prev_op = ""
    op_count = 1
    worksheet.write(1, 3, f"Data directory: {dir}", cell_format)
    dataset_count = 0
    for data_series in all_data:
        if prev_op != data_series.op:
            prev_op = data_series.op
            pad_top = 1 + 10*(op_count)
            pad_left = 2
            dataset_count = 0
            op_count = op_count + 1

            worksheet.write(pad_top, pad_left+1, f"{data_series.op}", cell_format)
            worksheet.write(pad_top+1, pad_left, f"num-gpus", cell_format)
            worksheet.write(pad_top+1, pad_left+1, f"num-wgs", cell_format)
            worksheet.write(pad_top+1, pad_left+2, f"num-threads", cell_format)
            for i in range(0, rows, 1):
                worksheet.write(pad_top+1, i+pad_left+3, x_str[i], cell_format)

        top_start = pad_top+2 + dataset_count
        worksheet.write(top_start, pad_left, f"{data_series.ngpus}", cell_format)
        worksheet.write(top_start, pad_left+1, f"{data_series.nwgs}", cell_format)
        worksheet.write(top_start, pad_left+2, f"{data_series.nthreads}", cell_format)

        for i in range(0, rows, 1):
            msg_size = x[i]
            data_val = None
            for pt in data_series.data:
                if pt.volume == msg_size:
                    data_val = pt.avg_time
                    break
            worksheet.write(top_start, i+pad_left+3, data_val, num_format)

        dataset_count = dataset_count + 1

workbook.close()
