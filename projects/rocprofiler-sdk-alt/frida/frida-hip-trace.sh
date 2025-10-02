#!/bin/bash

# Enhanced script to trace HIP API functions using Frida with readable arguments
# Usage: ./frida-hip-trace.sh [app_path] [app_args...]

# Default application path
DEFAULT_APP_PATH="/home/aelwazir/work/rocm-systems/projects/rocprofiler-sdk/build/bin/simple-transpose"
FRIDA_SCRIPT="/home/aelwazir/work/frida/frida-hip-tracer.js"
OUTPUT_FILE="/home/aelwazir/work/frida/hip-frida-trace.log"
CSV_FILE="/home/aelwazir/work/frida/hip-frida-trace.csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
if [ $# -eq 0 ]; then
    APP_PATH="$DEFAULT_APP_PATH"
    APP_ARGS=""
    echo -e "${BLUE}=== Frida HIP API Tracer ===${NC}"
    echo -e "${GREEN}Using default application:${NC} $APP_PATH"
elif [ $# -eq 1 ]; then
    APP_PATH="$1"
    APP_ARGS=""
    echo -e "${BLUE}=== Frida HIP API Tracer ===${NC}"
    echo -e "${GREEN}Application:${NC} $APP_PATH"
else
    APP_PATH="$1"
    shift
    APP_ARGS="$@"
    echo -e "${BLUE}=== Frida HIP API Tracer ===${NC}"
    echo -e "${GREEN}Application:${NC} $APP_PATH"
    echo -e "${GREEN}Arguments:${NC} $APP_ARGS"
fi

echo -e "${GREEN}Frida Script:${NC} $FRIDA_SCRIPT"
echo -e "${GREEN}Output File:${NC} $OUTPUT_FILE"
echo -e "${GREEN}CSV File:${NC} $CSV_FILE"
echo ""

# Check if Frida is installed
if ! command -v frida &> /dev/null && ! command -v ~/.local/bin/frida &> /dev/null; then
    echo -e "${RED}Error: Frida is not installed!${NC}"
    echo -e "${YELLOW}Please install Frida first:${NC}"
    echo "  pip install frida-tools"
    echo "  # or"
    echo "  pip install frida"
    exit 1
fi

# Check if the application exists
if [ ! -f "$APP_PATH" ]; then
    echo -e "${RED}Error: Application not found at $APP_PATH${NC}"
    echo -e "${YELLOW}Please check the path and ensure the application exists.${NC}"
    exit 1
fi

# Check if the Frida script exists
if [ ! -f "$FRIDA_SCRIPT" ]; then
    echo -e "${RED}Error: Frida script not found at $FRIDA_SCRIPT${NC}"
    echo -e "${YELLOW}Please ensure the frida-hip-tracer.js file exists.${NC}"
    exit 1
fi

# Check if the application is executable
if [ ! -x "$APP_PATH" ]; then
    echo -e "${YELLOW}Warning: Application is not executable. Making it executable...${NC}"
    chmod +x "$APP_PATH"
fi

# Clear previous output files
if [ -f "$OUTPUT_FILE" ] || [ -f "$CSV_FILE" ]; then
    echo -e "${YELLOW}Clearing previous trace files...${NC}"
    rm -f "$OUTPUT_FILE" "$CSV_FILE"
fi

echo -e "${GREEN}Starting HIP API tracing with Frida...${NC}"
echo -e "${BLUE}Press Ctrl+C to stop tracing${NC}"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Tracing stopped.${NC}"
    if [ -f "$OUTPUT_FILE" ] || [ -f "$CSV_FILE" ]; then
        echo -e "${GREEN}Trace results saved to:${NC}"
        echo -e "  Log file: $OUTPUT_FILE"
        echo -e "  CSV file: $CSV_FILE"
        echo ""
        echo -e "${BLUE}=== Quick Analysis ===${NC}"
        if [ -f "$OUTPUT_FILE" ]; then
            echo -e "${GREEN}Total function calls:${NC} $(grep -c "ENTER" "$OUTPUT_FILE" 2>/dev/null || echo "0")"
            echo -e "${GREEN}Unique functions called:${NC} $(grep "ENTER" "$OUTPUT_FILE" 2>/dev/null | cut -d' ' -f4 | cut -d'(' -f1 | sort -u | wc -l)"
            echo ""
            echo -e "${BLUE}=== Most Called Functions ===${NC}"
            grep "ENTER" "$OUTPUT_FILE" 2>/dev/null | cut -d' ' -f4 | cut -d'(' -f1 | sort | uniq -c | sort -nr | head -10
        fi
        echo ""
        echo -e "${BLUE}=== Usage Examples ===${NC}"
        echo "View full trace:"
        echo "  cat $OUTPUT_FILE"
        echo ""
        echo "View CSV data:"
        echo "  cat $CSV_FILE"
        echo "  # Import into Excel/Google Sheets for analysis"
        echo ""
        echo "Filter specific functions:"
        echo "  grep 'hipMalloc' $OUTPUT_FILE"
        echo "  grep 'hipMemcpy' $OUTPUT_FILE"
        echo "  grep 'hipLaunchKernel' $OUTPUT_FILE"
        echo ""
        echo "View function durations:"
        echo "  grep 'duration=' $OUTPUT_FILE | sort -k6 -nr"
        echo "  # Or from CSV: awk -F',' 'NR>1 {print \$3,\$7}' $CSV_FILE | sort -k2 -nr"
        echo ""
        echo "View by timestamp:"
        echo "  grep '2024-' $OUTPUT_FILE | head -20"
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run Frida with the HIP tracer
echo -e "${GREEN}Launching application with Frida...${NC}"
if command -v frida &> /dev/null; then
    if [ -n "$APP_ARGS" ]; then
        frida -l "$FRIDA_SCRIPT" -f "$APP_PATH" -- "$APP_ARGS"
    else
        frida -l "$FRIDA_SCRIPT" -f "$APP_PATH"
    fi
else
    if [ -n "$APP_ARGS" ]; then
        ~/.local/bin/frida -l "$FRIDA_SCRIPT" -f "$APP_PATH" -- "$APP_ARGS"
    else
        ~/.local/bin/frida -l "$FRIDA_SCRIPT" -f "$APP_PATH"
    fi
fi

# If we get here, the application finished normally
cleanup