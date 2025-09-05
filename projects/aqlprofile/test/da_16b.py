import sys

import text_sample_ex
import binary_sample_ex

#source file handling: from last command line argument or default
argc = len(sys.argv)
#print(argc)
#print(sys.argv)

#default source file
se_no = 0
has_exp_value = False
exp_value = 0
has_exp_range = False
exp_range = [0, 0]
is_text_file = True
length = -1

for i in range(1, argc):
    #print(sys.argv[i])
    if sys.argv[i] == "-h":
        print("help info")
        print("python3 da_16b.py [options] file_name")
        print("  -h:                     Print help info")
        print("  -s size:                Specify valid lenght of SPM samples in a binary data file")
        print("  -b:                     Specify file is a binary")
        print("  -e exp_val:             Specify expected value for the first data sample")
        print("  -r lower_val,upper_val: Specify range of the first data sample")
        sys.exit(0)
    if sys.argv[i] == "-se":
        se_no = int(sys.argv[i + 1])
        if se_no > 3:
            se_no = 0
        i += 1
    if sys.argv[i] == "-e":
        has_exp_value = True
        exp_value =  int(sys.argv[i + 1])
        i += 1
    if sys.argv[i] == "-r":
        has_exp_range = True
        exp_range = sys.argv[i + 1].split(',')
        i += 1
    if sys.argv[i] == "-b":
        is_text_file = False
    if sys.argv[i] == "-s":
        length = int(sys.argv[i + 1])
        i += 1

#spm_source = "/home/chun/proj/aqlprofile_spm_new/build/test/spm_dump_0.txt"
spm_source = sys.argv[argc - 1]

message = "Info: data analysis for "
if is_text_file:
    message += "text sample file: "
else:
    message += "binary sample file: "
    message += "with lenght " + str(length) + "B "

message += spm_source

if has_exp_value:
    message += "\n\twith expect data: " + str(exp_value)
if has_exp_range:
    message += "\n\twith expect data range: " + str(exp_range[0]) + " - " + str(exp_range[1])

print(message)

if is_text_file:
    text_sample_ex.sample_ex(spm_source)
else:
    binary_sample_ex.sample_ex(spm_source, length)

print("done sample extraction!")

print("extract counter samples for se%d"%(se_no))

file = open("samples.txt", "r")
lines = file.readlines()
file.close()


tss = []
se1s = []
se2s = []
se3s = []
se4s = []

#global time stamps
index = 0
delta = 0
prev_time = 0
start_ts = ""

while index < len(lines):
    line = lines[index + 3].strip() + lines[index + 2].strip() + lines[index + 1].strip() + lines[index].strip()
    aligned_index = '[%4s]' % (str(index))
    if index == 0:
        tss.append(line + aligned_index + ",  -")
        start_ts = line
    else:
        delta = int(line, 16) - prev_time
        tss.append(line + aligned_index + "," + str(delta))
    prev_time = int(line, 16)
    index += 160

#print(start_ts)
#print(delta)

print("total spm sample(s): %d"%(len(tss)))

#shader engine 1 samples
sum0 = 0
sum1 = 0
sum2 = 0
sum3 = 0
sum4 = 0
sum5 = 0
sum6 = 0
sum7 = 0
sum8 = 0
sum9 = 0
sum10 = 0
sum11 = 0
sum12 = 0
sum13 = 0
sum14 = 0
sum15 = 0

index = 32 * (se_no + 1)
while index < len(lines):
    aligned_index = '[%4s]' % (str(index))
    event0 = lines[index].strip() + aligned_index
    sum0 += int(lines[index].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16))
    event1 = lines[index + 16].strip() + aligned_index
    sum1 += int(lines[index + 16].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 1))
    event2 = lines[index + 1].strip() + aligned_index
    sum2 += int(lines[index + 1].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 1))
    event3 = lines[index + 16 + 1].strip() + aligned_index
    sum3 += int(lines[index + 16 + 1].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 2))
    event4 = lines[index + 2].strip() + aligned_index
    sum4 += int(lines[index + 2].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 2))
    event5 = lines[index + 16 + 2].strip() + aligned_index
    sum5 += int(lines[index + 16 + 2].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 3))
    event6 = lines[index + 3].strip() + aligned_index
    sum6 += int(lines[index + 3].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 3))
    event7 = lines[index + 16 + 3].strip() + aligned_index
    sum7 += int(lines[index + 16 + 3].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 4))
    event8 = lines[index + 4].strip() + aligned_index
    sum8 += int(lines[index + 4].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 4))
    event9 = lines[index + 16 + 4].strip() + aligned_index
    sum9 += int(lines[index + 16 + 4].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 5))
    event10 = lines[index + 5].strip() + aligned_index
    sum10 += int(lines[index + 5].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 5))
    event11 = lines[index + 16 + 5].strip() + aligned_index
    sum11 += int(lines[index + 16 + 5].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 6))
    event12 = lines[index + 6].strip() + aligned_index
    sum12 += int(lines[index + 6].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 6))
    event13 = lines[index + 16 + 6].strip() + aligned_index
    sum13 += int(lines[index + 16 + 6].strip(), 16)

    aligned_index = '[%4s]' % (str(index + 7))
    event14 = lines[index + 7].strip() + aligned_index
    sum14 += int(lines[index + 7].strip(), 16)
    aligned_index = '[%4s]' % (str(index + 16 + 7))
    event15 = lines[index + 16 + 7].strip() + aligned_index
    sum15 += int(lines[index + 16 + 7].strip(), 16)

    line = event0 + "," + event1 + "," + event2 + "," + event3 + "," + event4 + "," + event5 + "," + event6 + "," + event7 + "," + event8 + "," + event9+ "," + event10 + "," + event11 + "," + event12 + "," + event13 + "," + event14 + "," + event15
    se1s.append(line)
    index += 160
sums = str(sum0) + "," + str(sum1) + "," + str(sum2) + "," + str(sum3) + "," + str(sum4) + "," + str(sum5) + "," + str(sum6) + "," + str(sum7) + "," + str(sum8) + "," + str(sum9) + "," + str(sum10) + "," + str(sum11) + "," + str(sum12) + "," + str(sum13) + "," + str(sum14) + "," + str(sum15) 

def save_samples(output_file, tss, se1s):
    file = open(output_file, "w")
    size = len(tss)
    se_empty = False
    if len(se1s) == 0:
        se_empty = True
    #print("xxx: se_empty %r\n"%(se_empty))
    #print("tss: %d, se1s:%d"%(len(tss), len(se1s)))
    for i in range(0, size):
        if se_empty:
            file.write("%d,%s\n"%(i,tss[i]))
        else:
            file.write("%d,%s,%s\n"%(i,tss[i], se1s[i]))
    file.write("%s\n"%(sums))
    file.close()

output_file = "ofs.txt"

save_samples(output_file, tss, se1s)

print("Info: Time stamp starts at %s with delta %d"%(start_ts, delta))


if has_exp_value:
    if exp_value == sum0:
        print("Passed: Sample data %d matches expected data"%(sum0))
    else:
        print("Failed: Sample data %d does not match expected data %d"%(sum0, exp_value))

elif has_exp_range:
    if (int(exp_range[0]) <= sum0) and (sum0 <= int(exp_range[1])):
        print("Passed: Sample data %d is within expected data range [%s, %s]"%(sum0, exp_range[0], exp_range[1]))
    else:
        print("Failed: Sample data %d falls out side of expected data range [%s, %s]"%(sum0, exp_range[0], exp_range[1]))
else:
    print("Passed: Sample data %d, Please check file ofs.txt for detail"%(sum0))
