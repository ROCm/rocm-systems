import sys
import math
import statistics
#from matplotlib import pyplot as plt
import xlsxwriter
from datetime import datetime

f1 = sys.argv[1]

## EDIT THESE VALUES
bkc_version = "BKC 00.25.11.02 (38.4 GT/s XGMI)"
ifwi_version = "IFWI 00.940.956978"
amdgpu_version = "amdgpu 6.14.14-2193512"
os_kernel = "CentOS Kernel 6.9.0-0_fbk10_brcmrdma13_141_g9b20106afb70"
nic_driver = "Broadcom driver=6.9.0-0_fbk10_brcmrdma13_141_g9 firmware=232.0.213.0/pkg 232.1.190.0"
###

rccl_version = ""
rocm_version = ""
hip_version = ""

search_term = "(elements)"

file_prefixes = [f1, f1, f1, f1, f1, f1]
file_names = [
    "get_n2_w1_z16_4194304B.log",
    "wgget_n2_w16_z64_4194304B.log",
    "waveget_n2_w16_z128_4194304B.log",
    "put_n2_w1_z16_4194304B.log",
    "wgput_n2_w16_z64_4194304B.log",
    "waveput_n2_w16_z128_4194304B.log",
]
files = len(file_prefixes)

unique="rocSHMEM_MI300_Thor2_Heatmap"
dt="11-20-2025_v1"

## no. of nodes
nnodes = [2]
nodes = len(nnodes)

## cols => no. of consecutive runs
## rows => no. of msg sizes in the sweep -- 35 = 1B-16GB
cols, rows = 1, 19

#x = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824, 2147483648, 4294967296]
x = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]
#x_str = [16, 32, 64, 128, 256, 512, '1KB', '2KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB', '2GB', '4GB']
x_str = [16, 32, 64, 128, 256, 512, '1KB', '2KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB']


size1 = [[["" for _ in range(rows)] for _ in range(nodes)] for _ in range(files)]
msgcount1 = [[["" for _ in range(rows)] for _ in range(nodes)] for _ in range(files)]

avg_time1 = [[[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(nodes)] for _ in range(files)]
avg_bw1 = [[[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(nodes)] for _ in range(files)]
avg_msg_rate1 = [[[[0 for _ in range(cols)] for _ in range(rows)] for _ in range(nodes)] for _ in range(files)]


#colls = ['get', 'wgget', 'waveget', 'put', 'wgput', 'waveput', 'alltoall']
colls = ['get', 'wgget', 'waveget', 'put', 'wgput', 'waveput']

#uniq = f"{unique}_{dt}" maybe use date and include ipc vs. gda
uniq = f"rocshmem_test"
workbook = xlsxwriter.Workbook(f"{uniq}.xlsx")
workbook.set_properties({'company':  'AMD'})

merge_format1 = workbook.add_format({'bold':     True,
                                    'border':    5,
                                    'align':     'center',
                                    'valign':    'vcenter',
                                    'text_wrap': True,
                                    'fg_color':  '#FFFF00'})

merge_format2 = workbook.add_format({'bold':      True,
                                    'border':     5,
                                    'align':      'center',
                                    'valign':     'vcenter',
                                    'text_wrap':  True,
                                    'fg_color':   '#FFFF00',
                                    'font_color': '#FF0000'})

merge_format3 = workbook.add_format({'bold':     True,
                                    'border':    5,
                                    'align':     'center',
                                    'valign':    'vcenter',
                                    'text_wrap': True})

merge_format4 = workbook.add_format({'bold':     True,
                                    'align':     'center',
                                    'valign':    'vcenter',
                                    'text_wrap': True,
                                    'top':       5,
                                    'bottom':    5})

merge_format5 = workbook.add_format({'bold':     True,
                                    'align':     'center',
                                    'valign':    'vcenter',
                                    'text_wrap': True,
                                    'top':       5,
                                    'bottom':    5,
                                    'left':      5})

merge_format6 = workbook.add_format({'bold':     True,
                                    'align':     'center',
                                    'valign':    'vcenter',
                                    'text_wrap': True,
                                    'top':       5,
                                    'bottom':    5,
                                    'right':     5})

cell_format1 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'top': 5, 'left': 5})
cell_format2 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'top': 5})
cell_format3 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'top': 5, 'right': 5})
cell_format4 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'left': 5})
cell_format5 = workbook.add_format({'align': 'center', 'valign': 'vcenter'})
cell_format6 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'right': 5})
cell_format7 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bottom': 5, 'left': 5})
cell_format8 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bottom': 5})
cell_format9 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bottom': 5, 'right': 5})

cell_format_high4 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'left': 5, 'bold': True, 'fg_color': '#FFFF00'})
cell_format_high5 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True, 'fg_color': '#FFFF00'})
cell_format_high6 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bold': True, 'right': 5, 'fg_color': '#FFFF00'})

num_format1 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'top': 5, 'left': 5, 'num_format': '0.00'})
num_format2 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'top': 5, 'num_format': '0.00'})
num_format3 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'top': 5, 'right': 5, 'num_format': '0.00'})
num_format4 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'left': 5, 'num_format': '0.00'})
num_format5 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': '0.00'})
num_format6 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'right': 5, 'num_format': '0.00'})
num_format7 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bottom': 5, 'left': 5, 'num_format': '0.00'})
num_format8 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bottom': 5, 'num_format': '0.00'})
num_format9 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'bottom': 5, 'right': 5, 'num_format': '0.00'})

num_format_high4 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'left': 5, 'num_format': '0.00', 'bold': True, 'fg_color': '#FFFF00'})
num_format_high5 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'num_format': '0.00', 'bold': True, 'fg_color': '#FFFF00'})
num_format_high6 = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'right': 5, 'num_format': '0.00', 'bold': True, 'fg_color': '#FFFF00'})

now = datetime.now()
date_str = now.strftime("%Y-%m-%d");
worksheet = workbook.add_worksheet(f"{date_str}")
worksheet.set_zoom(70)

for fprefix in range(0, files, 1):
    coll = colls[fprefix]
       
    filename_prefix = file_prefixes[fprefix]

    for n in range(1, nodes+1):
        mpirun_cmd = ""

        filename=f"{filename_prefix}/{file_names[fprefix]}"
        print(filename)

        with open(f"{filename}", 'r') as file1:
            for lno1, line1 in enumerate(file1, start=0):
                if "mpirun " in line1.rstrip():
                    mpirun_cmd = line1.rstrip()
                    print(mpirun_cmd)
                    continue

                if "#" in line1.rstrip():
                    continue

                values = line1.rstrip().split()
                print(values)

                size1[fprefix][n-1][lno1-3] = int(values[0])
                msgcount1[fprefix][n-1][lno1-3] = int(values[1])
                avg_time1[fprefix][n-1][lno1-3][0] = float(values[2])
                avg_bw1[fprefix][n-1][lno1-3][0] = float(values[3])
                avg_msg_rate1[fprefix][n-1][lno1-3][0] = float(values[4])

            print(f"{coll} nodes: {nnodes[n-1]} {fprefix}")

    for fprefix in range(0, files, 1): 
        pad_top = 1 + 10*(fprefix)
        pad_left = 2

        worksheet.write(pad_top, pad_left+1, f"{file_names[fprefix]}", cell_format5)
        worksheet.write(pad_top+1, pad_left, "size (H)", cell_format5)
        for i in range(0, rows, 1):
            worksheet.write(pad_top+1, i+pad_left+1, x_str[i], cell_format5)

        for n in range(1, nodes+1, 1):
            top_start = pad_top+1 + n
            worksheet.write(top_start, pad_left, f"{nnodes[n-1]}-GPUs", cell_format5)
            for i in range(0, rows, 1):
                worksheet.write(top_start, i+pad_left+1, float(avg_time1[fprefix][n-1][i][0]), num_format5)
workbook.close()
