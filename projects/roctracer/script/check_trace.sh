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

tracename=$1
trace="${tracename}.txt"
rtrace="${tracename}_r.txt"
label=$2
trace_level=0

eval_trlevel() {
  tracefilename=$(basename $1)
  trace_level=`grep -w $tracefilename tests_trace_cmp_levels.txt | awk -F' ' '{print $2}'`
  echo "Trace comparison for $tracefilename is at level $trace_level"
}

eval_trlevel $tracename

case "$trace_level" in
  "0") echo "$label: PASSED"
  ;;
  "1") 
    #echo "../script/parse_trace.py -in $rtrace  -cn"
    echo "parse_trace.gen_events_info($rtrace,cn=1,or=0)"
    cnt_r=$(python -c 'import sys;sys.path.append("../script");import parse_trace; print parse_trace.gen_events_info("'$rtrace'",1,0)')
    #cnt_r=`../script/parse_trace.py -in $rtrace  -cn`
    #cnt=`../script/parse_trace.py -in $trace  -cn`
    cnt=$(python -c 'import sys;sys.path.append("../script");import parse_trace; print parse_trace.gen_events_info("'$trace'",1,0)')
    if [ "$cnt_r" = "$cnt" ] ; then
      echo "$label: PASSED (trace compare, events count)"
    else
      echo "$label: FAILED (trace compare, events count)"
      test_status=$(($test_status + 1))
    fi
  ;;
  "2")
    #echo "../script/parse_trace.py -in $rtrace  -or"
    echo "parse_trace.gen_events_info($rtrace,cn=0,or=1)"
    cnt_r=$(python -c 'import sys;sys.path.append("../script");import parse_trace; print parse_trace.gen_events_info("'$rtrace'",0,1)')
    #cnt_r=`../script/parse_trace.py -in $rtrace  -or`
    #cnt=`../script/parse_trace.py -in $trace  -or`
    cnt=$(python -c 'import sys;sys.path.append("../script");import parse_trace; print parse_trace.gen_events_info("'$trace'",0,1)')
    if [ "$cnt_r" = "$cnt" ] ; then
      echo "$label: PASSED (trace compare, events order)"
    else
      echo "$label: FAILED (trace compare, events order)"
      test_status=$(($test_status + 1))
    fi
  ;; 
  "3")
   echo "Comparing $trace $rtrace :"
   eval "diff --brief $trace $rtrace"
   if [ $? != 0 ] ; then
     echo "$label: FAILED (trace compare, files differ)"
     test_status=$(($test_status + 1))
   else
     echo "$label: PASSED (trace compare, files are the same)"
   fi
  ;;
esac
