#!/bin/bash
# Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
# Parallel topology visualization - renders channels concurrently

DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
FORMAT="svg"
DPI=300
JOBS=0  # 0 = auto-detect CPU count
KEEP_TEMP=false
MERGE=false

exit_error() {
  echo "Usage: $0 -i input_filename [-f format] [-d dpi] [-j jobs] [-m] [-k]"
  echo ""
  echo "Parallel topology visualizer for large graphs (256+ GPUs)"
  echo ""
  echo "Options:"
  echo "  -i input_filename   Input log file (required)"
  echo "  -f format           Output format: svg (default), png, pdf"
  echo "  -d dpi              DPI for PNG output (default: 300)"
  echo "  -j jobs             Number of parallel jobs (default: auto = CPU count)"
  echo "  -m                  Merge all channels into single output file"
  echo "  -k                  Keep temporary files (for debugging)"
  echo ""
  echo "Output:"
  echo "  Without -m: Creates <input>_<channel>.<format> for each channel"
  echo "  With -m:    Creates <input>.<format> with all channels merged"
  echo ""
  echo "Examples:"
  echo "  $0 -i large_topo.log -j 16              # 16 parallel jobs, separate SVGs"
  echo "  $0 -i large_topo.log -f png -d 600 -m   # Merged high-DPI PNG"
  exit 1
}

while getopts ":i:f:d:j:mkh" options; do
  case "${options}" in
    i) INPUT_NAME=${OPTARG} ;;
    f) FORMAT=${OPTARG}
       if [[ ! "$FORMAT" =~ ^(svg|png|pdf)$ ]]; then
         echo "Error: Invalid format '$FORMAT'. Supported: svg, png, pdf"
         exit 1
       fi ;;
    d) DPI=${OPTARG} ;;
    j) JOBS=${OPTARG} ;;
    m) MERGE=true ;;
    k) KEEP_TEMP=true ;;
    h) exit_error ;;
    :) echo "Error: -${OPTARG} requires an argument."; exit_error ;;
    ?) exit_error ;;
  esac
done

if [ -z "$INPUT_NAME" ]; then
  exit_error
fi

if [ ! -f "$INPUT_NAME" ]; then
  echo "Error: Input file '$INPUT_NAME' not found"
  exit 1
fi

# Auto-detect job count
if [ "$JOBS" -eq 0 ]; then
  JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
fi

# Create temp directory
TMPDIR=$(mktemp -d -t topo_parallel.XXXXXX)
trap '[ "$KEEP_TEMP" = false ] && rm -rf "$TMPDIR"' EXIT

echo "Processing topology with $JOBS parallel jobs..."
START_TIME=$(date +%s.%N)

# Step 1: Extract and split into separate DOT files
echo "Step 1/3: Extracting topology channels..."
export TOPO_OUTDIR="$TMPDIR"
$DIR/extract_topo_split.awk "$INPUT_NAME"

if [ ! -f "$TMPDIR/manifest.txt" ]; then
  echo "Error: Failed to extract topology"
  exit 1
fi

CHANNEL_COUNT=$(wc -l < "$TMPDIR/manifest.txt")
echo "  Found $CHANNEL_COUNT channels"

# Step 2: Render DOT files in parallel
echo "Step 2/3: Rendering channels in parallel..."

DOT_OPTS=""
if [ "$FORMAT" = "png" ]; then
  DOT_OPTS="-Gdpi=$DPI"
fi

# Create render script for parallel execution
cat > "$TMPDIR/render.sh" << RENDERSCRIPT
#!/bin/bash
DOTFILE="\$1"
OUTFILE="\${DOTFILE%.dot}.$FORMAT"
ERRFILE="\${DOTFILE%.dot}.err"

if ! dot -T$FORMAT $DOT_OPTS "\$DOTFILE" -o "\$OUTFILE" 2>"\$ERRFILE"; then
  echo "Warning: dot failed for '\$(basename \$DOTFILE)'. Error: \$(cat \$ERRFILE)" >&2
fi
RENDERSCRIPT
chmod +x "$TMPDIR/render.sh"

# Find all DOT files and render in parallel
find "$TMPDIR" -name "*.dot" | xargs -P "$JOBS" -I {} "$TMPDIR/render.sh" {}

# Step 3: Output results
echo "Step 3/3: Generating output..."

BASENAME=$(basename "$INPUT_NAME")
OUTDIR=$(dirname "$INPUT_NAME")

if [ "$MERGE" = true ]; then
  # Merge all rendered outputs (for SVG, combine; for PNG/PDF, create montage)
  OUTPUT_FILE="$OUTDIR/${BASENAME}.$FORMAT"

  if [ "$FORMAT" = "svg" ]; then
    # Create combined SVG with embedded sub-SVGs
    $DIR/merge_svg.sh "$TMPDIR" "$OUTPUT_FILE"
  else
    # For PNG/PDF, use montage if available, otherwise just copy first
    if command -v montage &> /dev/null; then
      montage "$TMPDIR"/*.$FORMAT -tile 1x -geometry +0+0 "$OUTPUT_FILE"
    else
      echo "Warning: montage (ImageMagick) not found, copying individual files"
      MERGE=false
    fi
  fi

  if [ "$MERGE" = true ]; then
    echo "Created merged output: $OUTPUT_FILE"
  fi
fi

if [ "$MERGE" = false ]; then
  # Copy individual files to output directory
  for f in "$TMPDIR"/*.$FORMAT; do
    if [ -f "$f" ]; then
      CHANNEL=$(basename "$f" .$FORMAT)
      cp "$f" "$OUTDIR/${BASENAME}_${CHANNEL}.$FORMAT"
    fi
  done
  echo "Created $CHANNEL_COUNT output files: ${BASENAME}_<channel>.$FORMAT"
fi

END_TIME=$(date +%s.%N)
TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
echo "Completed in ${TOTAL_TIME}s"

if [ "$KEEP_TEMP" = true ]; then
  echo "Temporary files kept in: $TMPDIR"
fi

exit 0
