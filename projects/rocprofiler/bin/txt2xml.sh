#!/bin/bash

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

timestamp=`date +%y%m%d_%H%M%S`

if [ $# = 0 ] ; then
  echo "Usage: $0 <input text file> [output dir]"
  exit -1
fi

input=$1
outdir=$2
if [ -z "$outdir" ] ; then
  outdir="."
fi

range=""
kernel=""
gpu_index=""

parse() {
  scan="$1"
  index=0
  while read -r line || [[ -n "$line" ]] ; do
    line=`echo $line | sed "s/\s*#.*$//"`
    if [ -z "$line" ] ; then
      continue
    fi

    feature=`echo $line | sed -n "s/^\s*\([a-z]*\)\s*:.*$/\1/p"`
    line=`echo $line | sed "s/^[^:]*:\s*//"`
    line=`echo "$line" | sed -e "s/\s*=\s*/=/g" -e "s/\s*:\s*/:/g" -e "s/,\{1,\}/ /g" -e "s/\s\{1,\}/ /g" -e "s/\s*$//"`

    if [ "$scan" = 0 ] ; then
      line=`echo "$line" | sed -e "s/ /,/g"`
      if [ "$feature" == "range" ] ; then
        range=$line
      fi
      if [ "$feature" == "kernel" ] ; then
        kernel=$line
      fi
      if [ "$feature" == "gpu" ] ; then
        gpu_index=$line
      fi
    else
      found=$(echo $feature | sed -n "/^\(pmc\|hip\|hsa\)$/ p")
      if [ -n "$found" ] ; then
        output=$outdir/input${index}.xml
        header="# $timestamp '$output' generated with '$0 $*'"
        echo $header > $output

        if [ "$feature" == "pmc" ] ; then
          line=`echo "$line" | sed -e "s/ /,/g"`
          cat >> $output <<EOF
<metric range="$range" kernel="$kernel" gpu_index="$gpu_index"></metric>
<metric name=$line ></metric>
EOF
        fi

        if [ "$feature" == "hip" ] ; then
          cat >> $output <<EOF
<trace name="HIP"><parameters api="$line"></parameters></trace>
EOF
        fi

        if [ "$feature" == "hsa" ] ; then
          cat >> $output <<EOF
<trace name="HSA"><parameters api="$line"></parameters></trace>
EOF
        fi

      fi
    fi

    index=$((index + 1))
  done < $input
}

parse 0
parse 1

exit 0
