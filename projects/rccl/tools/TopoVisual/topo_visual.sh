#!/bin/bash
# Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exit_error() {
  echo "Usage: $0 [ -i input_filename ]"
  exit 1
}

while getopts ":i:o:" options; do
  case "${options}" in
    i)
      INPUT_NAME=${OPTARG}
      ;;
    :)
      echo "Error: -${OPTARG} requires an argument."
      exit_error
      ;;
    ?)
      exit_error
      ;;
  esac
done

if [ -z "$INPUT_NAME" ]
then
  exit_error
else
  $DIR/extract_topo.awk $INPUT_NAME | dot -Tpng -o "$INPUT_NAME.png"
  echo "Extracted topology from $INPUT_NAME to $INPUT_NAME.png"
fi

exit 0
