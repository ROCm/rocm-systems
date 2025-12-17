#!/bin/bash

# Directory path to search for JSON files
directory="../../build/release"

if command -v pip &>/dev/null; then
    echo "pip is already installed."
else
    echo "pip is not installed. Installing..."
    sudo apt-get update
    sudo apt install python3-pip
fi

required_library='pandas'

# Check if pandas is installed
if python3 -c "import $required_library" &> /dev/null; then
    echo "$required_library is already installed."
else
    echo "$required_library is not installed. Installing..."
    pip3 install $required_library
fi

required_library='plotly'

# Check if the library is installed
if python3 -c "import $required_library" &> /dev/null; then
    echo "$required_library is already installed."
else
    echo "$required_library is not installed. Installing..."
    pip3 install $required_library
fi

# Check if the file exists
if [ ! -f "$directory/.ninja_log" ]; then
  echo "File '$directory/.ninja_log' does not exist."
  exit 1
fi

declare -A unique_values

# Use awk to compare and delete duplicates
awk '!unique_values[$5]++' "$directory/.ninja_log" > temp_file.txt
mv temp_file.txt "$directory/.ninja_log"

# Rename the file with .csv extension
mv "$directory/.ninja_log" "$directory/time_trace.log"

# Run the python program
python3 time_trace_generator.py --min_val 5 --include_linking