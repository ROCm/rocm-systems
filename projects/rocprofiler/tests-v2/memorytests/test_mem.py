import sys

file = open(sys.argv[1], 'r')
Lines = file.readlines()

count = 0
flag = 0

print("Leaks Detected:")

for line in Lines:
  if "object(s) allocated from" in line:
    flag = 0
  if "rocprofiler/src" in line and flag == 0:
    print(line)
    count+=1
    flag = 1
    
if count == 0:
  print("No Leaks were found!")
else:
  print("Warning: Found (" + str(count) + ") Memory Leaks related to rocprofiler project!")