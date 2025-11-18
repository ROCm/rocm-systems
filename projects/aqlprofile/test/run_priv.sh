#!/bin/sh

# MIT License
# Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

RPATH=`realpath $0`

tbin=./ctrl
export LD_LIBRARY_PATH=$PWD

cd `dirname $RPATH`

echo "Run with PMC SCAN"
export AQLPROFILE_PMC=1
unset AQLPROFILE_PMC_PRIV
unset AQLPROFILE_SQTT
unset AQLPROFILE_SDMA
export AQLPROFILE_SCAN=1
unset AQLPROFILE_SPM
eval $tbin

echo "Run with SDMA SETUP Mode"
unset AQLPROFILE_PMC
unset AQLPROFILE_PMC_PRIV
unset AQLPROFILE_SQTT
export AQLPROFILE_SDMA=1
unset AQLPROFILE_SCAN
unset AQLPROFILE_SPM
eval $tbin

echo "Run with PMC Privilge"
unset AQLPROFILE_PMC
export AQLPROFILE_PMC_PRIV=1
unset AQLPROFILE_SQTT
unset AQLPROFILE_SDMA
unset AQLPROFILE_SCAN
unset AQLPROFILE_SPM
eval $tbin

#echo "Run with SPM"
#unset AQLPROFILE_PMC
#unset AQLPROFILE_PMC_PRIV
#unset AQLPROFILE_SQTT
#unset AQLPROFILE_SDMA
#unset AQLPROFILE_SCAN
#export AQLPROFILE_SPM=1
#eval $tbin

exit 0
