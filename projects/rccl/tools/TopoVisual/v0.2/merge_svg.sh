#!/bin/bash
# Merge multiple SVG files into a single SVG with stacked layout
# Usage: merge_svg.sh <input_dir> <output_file>

INPUT_DIR="$1"
OUTPUT_FILE="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_FILE" ]; then
  echo "Usage: $0 <input_dir> <output_file>"
  exit 1
fi

# Collect all SVG files sorted by name
SVG_FILES=$(find "$INPUT_DIR" -name "*.svg" | sort)
FILE_COUNT=$(echo "$SVG_FILES" | wc -l)

if [ "$FILE_COUNT" -eq 0 ]; then
  echo "No SVG files found in $INPUT_DIR"
  exit 1
fi

# Calculate total dimensions
TOTAL_HEIGHT=0
MAX_WIDTH=0
Y_OFFSET=0
PADDING=20

# First pass: calculate dimensions
declare -a HEIGHTS
declare -a WIDTHS
i=0
for svg in $SVG_FILES; do
  # Extract width and height from SVG (POSIX-compatible)
  W=$(sed -n 's/.*width="\([0-9][0-9]*\)".*/\1/p' "$svg" | head -n 1)
  H=$(sed -n 's/.*height="\([0-9][0-9]*\)".*/\1/p' "$svg" | head -n 1)
  
  # Default if not found
  [ -z "$W" ] && W=800
  [ -z "$H" ] && H=600
  
  WIDTHS[$i]=$W
  HEIGHTS[$i]=$H
  
  TOTAL_HEIGHT=$((TOTAL_HEIGHT + H + PADDING))
  [ "$W" -gt "$MAX_WIDTH" ] && MAX_WIDTH=$W
  
  ((i++))
done

# Create merged SVG
cat > "$OUTPUT_FILE" << HEADER
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="$MAX_WIDTH" height="$TOTAL_HEIGHT" viewBox="0 0 $MAX_WIDTH $TOTAL_HEIGHT">
  <style>
    .channel-group { }
  </style>
HEADER

# Second pass: embed each SVG
Y_OFFSET=0
i=0
for svg in $SVG_FILES; do
  CHANNEL=$(basename "$svg" .svg)
  
  # Extract the content between <svg> and </svg>, excluding the svg tags,
  # and stream it directly into the merged SVG to avoid storing large data
  # in a shell variable.
  {
    cat << CHANNEL_SVG_START
  <g class="channel-group" id="$CHANNEL" transform="translate(0, $Y_OFFSET)">
CHANNEL_SVG_START

    sed -n '/<svg/,/<\/svg>/p' "$svg" | sed '1d;$d' | sed 's/^/    /'

    cat << CHANNEL_SVG_END
  </g>
CHANNEL_SVG_END
  } >> "$OUTPUT_FILE"

  Y_OFFSET=$((Y_OFFSET + ${HEIGHTS[$i]} + PADDING))
  ((i++))
done

echo "</svg>" >> "$OUTPUT_FILE"

echo "Merged $FILE_COUNT SVG files into $OUTPUT_FILE"
