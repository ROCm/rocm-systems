#!/bin/sh

################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################

# paths to ROC profiler and oher libraries
export LD_LIBRARY_PATH=$PWD
# ROC profiler library loaded by HSA runtime
export HSA_TOOLS_LIB=librocprofiler64.so
# enable error messages logging to '/tmp/rocprofiler_log.txt'
export ROCPROFILER_LOG=1
# ROC profiler metrics config file
unset ROCP_PROXY_QUEUE
# ROC profiler metrics config file
export ROCP_METRICS=metrics.xml

eval ./test/standalone_test

# tool library loaded by ROC profiler
export ROCP_TOOL_LIB=libtool.so
# ROC profiler kernels timing
export ROCP_TIMESTAMP_ON=1
# output directory for the tool library, for metrics results file 'results.txt'
# and SQTT trace files 'thread_trace.se<n>.out'
export ROCP_OUTPUT_DIR=./RESULTS

if [ ! -e $ROCP_TOOL_LIB ] ; then
  export ROCP_TOOL_LIB=test/libtool.so
fi

export ROCP_KITER=100
export ROCP_DITER=100
export ROCP_INPUT=input.xml
eval ./test/ctrl

export ROCP_KITER=1
export ROCP_DITER=4
export ROCP_INPUT=input1.xml
eval ./test/ctrl

#valgrind --leak-check=full $tbin
#valgrind --tool=massif $tbin
#ms_print massif.out.<N>

exit 0
