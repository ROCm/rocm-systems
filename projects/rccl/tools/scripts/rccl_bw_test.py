#!/usr/bin/python3

# Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
import os
import argparse

print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print (' ')
print ('Usage example: python3 rccl_bw_test.py --np 2 --host_ip 10.67.10.104,10.67.10.154 --gpus_per_node 8')
print ('               --test_iteration 100 --test_exe /home/tmp1/rccl-tests/build/all_reduce_perf')
print (' ')
print ('Note: This script currently is only compatible with MPICH mpirun syntax')
print (' ')
print ("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


parser = argparse.ArgumentParser()

parser.add_argument('--np', type=int, default=1)
parser.add_argument('--host_ip', type=str, default=None)
parser.add_argument('--gpus_per_node', type=int, default=8)
parser.add_argument('--test_iteration', type=int, default=100)
parser.add_argument('--test_exe', type=str, default='/home/tmp1/rccl-tests/build/all_reduce_perf')
args = parser.parse_args()

nodes = str(args.np) + ' ' # number of mNodes
if (args.host_ip == None):
    host_ip = ''
else :
    host_ip = '-host ' + str(args.host_ip) + ' '
env1 = '-env HSA_FORCE_FINE_GRAIN_PCIE 1 '
env2 = '-env NCCL_DEBUG_INFO '
env3 = '-env NCCL_SOCKET_IFNAME enp98s0f0 '
env4 = '-env HIP_VISIBLE_DEVICES '
test_dir = str(args.test_exe) + ' '
test_param1 = '-b 8 ' # min size
test_param2 = '-e 128M ' # max size
test_param3 = '-f 2 '
test_param4 = '-g ' + str(args.gpus_per_node) + ' ' # number of GPUs


for i in range (1, args.test_iteration):
    arr = np.arange(8)
    np.random.shuffle(arr)
    hip_dev = ','.join(str(i) for i in arr)

    cmd = 'mpirun ' + '-np ' + nodes + host_ip + env1 + env3 + env4 + hip_dev + ' ' + test_dir + test_param1 + test_param2 + test_param3 + test_param4

    print(cmd)
    res = os.popen(cmd)
    output = res.read()
    print(output)
