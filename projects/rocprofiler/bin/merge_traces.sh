#!/bin/bash

################################################################################
# Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#A script to merge rocprof traces and then provide a results.json for the aggregate numbers.

BIN_DIR=$(dirname $(realpath ${BASH_SOURCE[0]}))

bin_name=`basename $0`

# usage method
usage() {
  echo "Script for aggregating results from multiple rocprofiler out directries."
  echo "Full path: $BIN_DIR/$bin_name"
  echo "Usage: if running independently"
  echo "  $bin_name -o <outputdir> [<inputdir>...]"
  echo ""
  echo "Usage: if running with rocprof"
  echo "  rocprof --merge-traces -o <outputdir> [<inputdir>...]"
  echo ""
  echo "Options:"
  echo "  -o <outputdir> - output directory where the results will be aggregated."
  echo "  <inputdir>... - space separated list of rocprofiler directories. If not specified, CWD is used."
  echo ""
  exit 1
}

# read arguments

INPUT_DIRS=()
while getopts "o:h" opt; do
  case $opt in
     o) OUTPUT_DIR=$OPTARG ;;
     h) usage ;;
     \?) usage ;;
esac
done
shift $((OPTIND-1))

INPUT_DIRS=$@

if [ "${OUTPUT_DIR}" = ""  ] ; then
  echo "Missing output dir option"
  usage
fi

for INPUT_DIR in ${INPUT_DIRS} ; do
  if [[ ! -d "${INPUT_DIR}" ]] ; then
    echo "Directory ${INPUT_DIR} does not exist."
    exit 1
  fi
done

if ! [ -d "${OUTPUT_DIR}" ] ; then
  mkdir -p "${OUTPUT_DIR}"
fi

echo "Processing directories: $INPUT_DIRS"
for file in begin_ts_file hcc_ops_trace hsa_handles hip_api_trace roctx_trace hsa_api_trace results async_copy_trace; do
  res=$(find ${INPUT_DIRS} -type f -regextype sed -regex ".*/[0-9]\{1,\}_${file}\.txt" \
	  -not -path "${OUTPUT_DIR}/*")
  test -n "${res}" && cat ${res} > "${OUTPUT_DIR}/${file}.txt"
done

if ! [ -d "${BIN_DIR}" ] ; then
  echo "Bin directory $BIN_DIR not found!"
  exit 1
fi

if [ -z "$ROCP_PYTHON_VERSION" ] ; then
  ROCP_PYTHON_VERSION=python3
fi

OUTPUT_LIST="$OUTPUT_DIR/results.txt"
db_output="$OUTPUT_DIR/results.db"
echo "$ROCP_PYTHON_VERSION $BIN_DIR/tblextr.py $db_output $OUTPUT_LIST"
$ROCP_PYTHON_VERSION $BIN_DIR/tblextr.py $db_output $OUTPUT_LIST
if [ "$?" -ne 0 ] ; then
  echo "Profiling data corrupted: '$OUTPUT_LIST'"
  exit 1
fi

