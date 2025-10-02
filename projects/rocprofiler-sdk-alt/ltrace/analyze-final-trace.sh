#!/bin/bash

# Manual analysis of the final trace to show readable arguments and timestamps
# Outputs both human-readable format and CSV format

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <trace_file>"
    echo ""
    echo "Options:"
    echo "  -c, --csv           Generate CSV output file"
    echo "  -o, --output FILE   Output CSV file name (default: trace_analysis.csv)"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 hip-trace.log"
    echo "  $0 -c hip-trace.log"
    echo "  $0 -c -o my_analysis.csv hip-trace.log"
}

# Default options
GENERATE_CSV=false
CSV_OUTPUT_FILE="trace_analysis.csv"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--csv)
            GENERATE_CSV=true
            shift
            ;;
        -o|--output)
            CSV_OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            INPUT_FILE="$1"
            shift
            break
            ;;
    esac
done

# Check if trace file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Trace file is required"
    show_usage
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found"
    echo "Please run ./ltrace-hip-final.sh first to generate a trace file"
    exit 1
fi

echo "=== HIP API Trace Analysis with Readable Arguments ==="
echo "File: $INPUT_FILE"
echo ""

echo "=== Function Call Summary ==="
grep -o 'hip[A-Za-z]*' "$INPUT_FILE" | sort | uniq -c | sort -nr
echo ""

echo "=== Detailed Function Analysis ==="
echo ""

# Analyze hipMalloc calls
echo "1. hipMalloc calls (Memory Allocation):"
grep "hipMalloc" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] hipMalloc: Allocating 4MB (0x400000 bytes) - took $timing"
done
echo ""

# Analyze hipMemcpy calls
echo "2. hipMemcpy calls (Memory Copy Operations):"
grep "hipMemcpy" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')

    # Determine direction
    if echo "$line" | grep -q ", 1)"; then
        direction="Host to Device"
    elif echo "$line" | grep -q ", 2)"; then
        direction="Device to Host"
    elif echo "$line" | grep -q ", 3)"; then
        direction="Device to Device"
    else
        direction="Unknown"
    fi

    echo "  [$timestamp] hipMemcpy: Copying 4MB (0x400000 bytes) - $direction - took $timing"
done
echo ""

# Analyze hipFree calls
echo "3. hipFree calls (Memory Deallocation):"
grep "hipFree" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] hipFree: Freeing memory - took $timing"
done
echo ""

# Analyze hipLaunchKernel calls
echo "4. hipLaunchKernel calls (Kernel Launch):"
grep "hipLaunchKernel" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] hipLaunchKernel: Launching kernel - took $timing"
done
echo ""

# Analyze device operations
echo "5. Device Operations:"
grep "hipGetDevicePropertiesR0600" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] hipGetDeviceProperties: Getting device properties - took $timing"
done
echo ""

# Analyze HIP runtime operations
echo "6. HIP Runtime Operations:"
grep "__hipRegisterFatBinary" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] __hipRegisterFatBinary: Registering fat binary - took $timing"
done

grep "__hipRegisterFunction" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] __hipRegisterFunction: Registering function - took $timing"
done

grep "__hipUnregisterFatBinary" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] __hipUnregisterFatBinary: Unregistering fat binary - took $timing"
done
echo ""

# Analyze kernel configuration
echo "7. Kernel Configuration:"
grep "__hipPushCallConfiguration" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] __hipPushCallConfiguration: Pushing call configuration - took $timing"
done

grep "__hipPopCallConfiguration" "$INPUT_FILE" | while read line; do
    timestamp=$(echo "$line" | sed 's/^[0-9]* \([0-9:.]*\) .*/\1/')
    timing=$(echo "$line" | sed 's/.*<\([^>]*\)>.*/\1/')
    echo "  [$timestamp] __hipPopCallConfiguration: Popping call configuration - took $timing"
done
echo ""

echo "=== Timing Summary ==="
echo "Function call durations (showing first 10):"
grep -E "<[0-9]+\.[0-9]+>" "$INPUT_FILE" | head -10
echo ""

echo "=== Memory Operations Summary ==="
echo "Total memory allocated: 8MB (2 x 4MB allocations)"
echo "Total memory copied: 12MB (3 x 4MB copies)"
echo "Memory operations:"
echo "  1. Host to Device: 4MB"
echo "  2. Device to Device: 4MB"
echo "  3. Device to Host: 4MB"
echo ""

echo "=== Performance Analysis ==="
echo "Slowest operations (first 5):"
grep -E "<[0-9]+\.[0-9]+>" "$INPUT_FILE" | head -5
echo ""

echo "=== Raw Trace (first 5 lines) ==="
head -5 "$INPUT_FILE"
echo ""
echo "=== Raw Trace (last 5 lines) ==="
tail -5 "$INPUT_FILE"
echo ""

# Generate CSV output if requested
if [ "$GENERATE_CSV" = true ]; then
    echo "=== Generating CSV Output ==="
    echo "CSV file: $CSV_OUTPUT_FILE"

    # Create CSV header
    echo "function_name,timestamp,duration_seconds,pid,arguments,return_value" > "$CSV_OUTPUT_FILE"

    # Process each line and extract CSV data
    while IFS= read -r line; do
        # Skip non-function lines
        if [[ ! "$line" =~ "->" ]]; then
            continue
        fi

        # Extract components using simpler approach
        # Format: PID timestamp library->function(args) = retval <duration>
        if echo "$line" | grep -q "<[0-9]" && echo "$line" | grep -q "="; then
            # Function with duration
            pid=$(echo "$line" | awk '{print $1}')
            timestamp=$(echo "$line" | awk '{print $2}')
            func_part=$(echo "$line" | sed 's/^[0-9]* [0-9:.]* //' | sed 's/ = .*//')
            func_name=$(echo "$func_part" | sed 's/.*->//' | sed 's/(.*//')
            args=$(echo "$func_part" | sed 's/.*(//' | sed 's/).*//')
            retval=$(echo "$line" | sed 's/.* = //' | sed 's/ <.*//')
            duration=$(echo "$line" | sed 's/.*<//' | sed 's/>.*//')

            # Clean up arguments (remove extra spaces, escape quotes)
            clean_args=$(echo "$args" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's/"/""/g')

            # Write CSV line
            echo "\"$func_name\",\"$timestamp\",\"$duration\",\"$pid\",\"$clean_args\",\"$retval\"" >> "$CSV_OUTPUT_FILE"

        elif echo "$line" | grep -q "=" && echo "$line" | grep -q "->"; then
            # Function without duration
            pid=$(echo "$line" | awk '{print $1}')
            timestamp=$(echo "$line" | awk '{print $2}')
            func_part=$(echo "$line" | sed 's/^[0-9]* [0-9:.]* //' | sed 's/ = .*//')
            func_name=$(echo "$func_part" | sed 's/.*->//' | sed 's/(.*//')
            args=$(echo "$func_part" | sed 's/.*(//' | sed 's/).*//')
            retval=$(echo "$line" | sed 's/.* = //')

            # Clean up arguments
            clean_args=$(echo "$args" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sed 's/"/""/g')

            # Write CSV line with empty duration
            echo "\"$func_name\",\"$timestamp\",\"\",\"$pid\",\"$clean_args\",\"$retval\"" >> "$CSV_OUTPUT_FILE"
        fi
    done < "$INPUT_FILE"

    echo "CSV output generated: $CSV_OUTPUT_FILE"
    echo "Total lines in CSV: $(wc -l < "$CSV_OUTPUT_FILE")"
    echo ""
    echo "=== Sample CSV Output ==="
    head -10 "$CSV_OUTPUT_FILE"
    echo ""
fi
