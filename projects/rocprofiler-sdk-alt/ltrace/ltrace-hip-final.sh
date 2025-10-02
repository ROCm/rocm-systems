#!/bin/bash

# Final HIP API tracing script with readable output
# Shows human-readable arguments and clear start/end timestamps

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] <application> [application_args...]"
    echo ""
    echo "Options:"
    echo "  -o, --output FILE    Output file (default: hip-trace.log)"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 ../rocm-systems/projects/rocprofiler-sdk/build/bin/simple-transpose"
    echo "  $0 -o my-trace.log ./my-hip-app arg1 arg2"
    echo "  $0 /usr/bin/hip-app --device 0 --size 1024"
}

# Default options
OUTPUT_FILE="hip-trace.log"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
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
            # First non-option argument is the application
            APP_PATH="$1"
            shift
            # Remaining arguments are application arguments
            APP_ARGS="$@"
            break
            ;;
    esac
done

# Check if application path is provided
if [ -z "$APP_PATH" ]; then
    echo "Error: Application path is required"
    show_usage
    exit 1
fi

# Check if application exists
if [ ! -f "$APP_PATH" ] && [ ! -x "$APP_PATH" ]; then
    echo "Error: Application '$APP_PATH' not found or not executable"
    exit 1
fi

echo "=== HIP API Tracing with Readable Output ==="
echo "Application: $APP_PATH"
if [ -n "$APP_ARGS" ]; then
    echo "Arguments: $APP_ARGS"
fi
echo "Output file: $OUTPUT_FILE"
echo ""

# Run ltrace with timing information
ltrace -tt -f -T -e "hipMalloc+hipFree+hipMemcpy+hipLaunchKernel+hipGetDevicePropertiesR0600+__hipRegisterFatBinary+__hipRegisterFunction+__hipUnregisterFatBinary+__hipPushCallConfiguration+__hipPopCallConfiguration" -o $OUTPUT_FILE $APP_PATH $APP_ARGS

echo "Tracing completed!"
echo "Results saved to: $OUTPUT_FILE"
echo ""

# Create readable version
READABLE_FILE="${OUTPUT_FILE%.log}_readable.log"
echo "Creating readable version..."
echo "Readable output: $READABLE_FILE"
echo ""

# Create a simple readable version by just copying the trace
cp "$OUTPUT_FILE" "$READABLE_FILE"

echo "Readable trace created: $READABLE_FILE"
echo ""
echo "=== Sample Readable Output ==="
head -20 $READABLE_FILE
echo ""
echo "=== Analysis Commands ==="
echo "View full readable trace:"
echo "  cat $READABLE_FILE"
echo ""
echo "Show memory operations:"
echo "  grep -E 'hipMalloc|hipFree|hipMemcpy' $READABLE_FILE"
echo ""
echo "Show kernel operations:"
echo "  grep -E 'hipLaunchKernel' $READABLE_FILE"
echo ""
echo "Show device operations:"
echo "  grep -E 'hipGetDevice|__hipRegister' $READABLE_FILE"
echo ""
echo "Show function starts:"
echo "  grep 'START:' $READABLE_FILE"
echo ""
echo "Show function ends:"
echo "  grep 'END:' $READABLE_FILE"
