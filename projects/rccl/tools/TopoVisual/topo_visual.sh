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

# Default values
FORMAT="svg"
DPI=300
LARGE_SCALE=false

exit_error() {
  echo "Usage: $0 -i input_filename [-f format] [-d dpi] [-l]"
  echo ""
  echo "Options:"
  echo "  -i input_filename   Input log file (required)"
  echo "  -f format           Output format: svg (default), png, pdf"
  echo "                      SVG recommended for large topologies (scalable, no blur)"
  echo "  -d dpi              DPI for PNG output (default: 300, use 600+ for large graphs)"
  echo "  -l                  Large graph mode: optimizes layout for 64+ GPUs"
  echo ""
  echo "Examples:"
  echo "  $0 -i nccl_log.txt                    # Generate SVG (recommended)"
  echo "  $0 -i nccl_log.txt -f png -d 600      # High-DPI PNG"
  echo "  $0 -i nccl_log.txt -f svg -l          # SVG with large graph optimizations"
  exit 1
}

while getopts ":i:f:d:lh" options; do
  case "${options}" in
    i)
      INPUT_NAME=${OPTARG}
      ;;
    f)
      FORMAT=${OPTARG}
      if [[ ! "$FORMAT" =~ ^(svg|png|pdf)$ ]]; then
        echo "Error: Invalid format '$FORMAT'. Supported: svg, png, pdf"
        exit 1
      fi
      ;;
    d)
      DPI=${OPTARG}
      if ! [[ "$DPI" =~ ^[0-9]+$ ]] || [ "$DPI" -lt 72 ]; then
        echo "Error: DPI must be a number >= 72"
        exit 1
      fi
      ;;
    l)
      LARGE_SCALE=true
      ;;
    h)
      exit_error
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

if [ -z "$INPUT_NAME" ]; then
  exit_error
fi

# Build dot command options
DOT_OPTS=""
if [ "$FORMAT" = "png" ]; then
  DOT_OPTS="-Gdpi=$DPI"
fi

# Large scale optimizations
LARGE_OPTS=""
if [ "$LARGE_SCALE" = true ]; then
  LARGE_OPTS="-Gsize=200,200! -Grankdir=TB -Gnodesep=0.5 -Granksep=1.0 -Gsplines=ortho"
fi

OUTPUT_FILE="$INPUT_NAME.$FORMAT"

# Generate the visualization
$DIR/extract_topo.awk $INPUT_NAME | dot -T$FORMAT $DOT_OPTS $LARGE_OPTS -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
  echo "Extracted topology from $INPUT_NAME to $OUTPUT_FILE"
  if [ "$FORMAT" = "svg" ]; then
    echo "Tip: SVG is a vector format - zoom in without blur in any web browser or image viewer"
  elif [ "$FORMAT" = "png" ] && [ "$DPI" -lt 300 ]; then
    echo "Tip: For large graphs, use higher DPI (-d 600) or switch to SVG format (-f svg)"
  fi
else
  echo "Error: Failed to generate visualization"
  exit 1
fi

exit 0
