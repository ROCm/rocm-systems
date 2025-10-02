# HIP API Tracing with ltrace

This directory contains scripts for tracing HIP API functions and their arguments with timestamps using `ltrace`.

## Files

- `ltrace-hip-final.sh` - Main tracing script with readable output and timing
- `analyze-final-trace.sh` - Analysis script for readable arguments and timestamps

## Quick Start

```bash
# Basic HIP API tracing
./ltrace-hip-final.sh /path/to/your/hip-application

# Analyze the generated trace (human-readable format)
./analyze-final-trace.sh hip-trace.log

# Generate CSV output for data analysis
./analyze-final-trace.sh -c hip-trace.log
```

## Usage

### ltrace-hip-final.sh

```bash
./ltrace-hip-final.sh [OPTIONS] <application> [application_args...]
```

**Options:**
- `-o, --output FILE` - Output file (default: hip-trace.log)
- `-h, --help` - Show help

**Examples:**
```bash
# Trace with default settings
./ltrace-hip-final.sh /home/aelwazir/work/rocm-systems/projects/rocprofiler-sdk/build/bin/hip-in-libraries

# Trace with custom output file
./ltrace-hip-final.sh -o my-trace.log ./my-hip-app arg1 arg2

# Trace with application arguments
./ltrace-hip-final.sh /usr/bin/hip-app --device 0 --size 1024
```

### analyze-final-trace.sh

```bash
./analyze-final-trace.sh [OPTIONS] <trace_file>
```

**Options:**
- `-c, --csv` - Generate CSV output file
- `-o, --output FILE` - Output CSV file name (default: trace_analysis.csv)
- `-h, --help` - Show help

**Examples:**
```bash
# Analyze the default trace file (human-readable format)
./analyze-final-trace.sh hip-trace.log

# Generate CSV output for data analysis
./analyze-final-trace.sh -c hip-trace.log

# Generate CSV with custom output file
./analyze-final-trace.sh -c -o my_analysis.csv hip-trace.log

# Analyze a custom trace file
./analyze-final-trace.sh my-trace.log
```

## Features

- **Timestamps**: Microsecond precision with `-tt` and `-T` for call durations
- **Function filtering**: Uses `+` syntax (e.g., `hipMalloc+hipFree+hipMemcpy`)
- **Child process tracing**: `-f` to follow child processes
- **HIP API coverage**: Includes public and internal HIP functions
- **Readable output**: Creates a readable version of the trace
- **Analysis tools**: Built-in commands for function counting, timing, and filtering

## Output Files

- `hip-trace.log` - Raw ltrace output with timestamps
- `hip-trace_readable.log` - Readable version of the trace
- `trace_analysis.csv` - CSV format with function names, timestamps, durations, and arguments

## Analysis Commands

After running the scripts, you can use these commands to analyze the traces:

```bash
# View function call summary
grep -o 'hip[A-Za-z]*' hip-trace.log | sort | uniq -c | sort -nr

# Show specific functions
grep 'hipMalloc' hip-trace.log
grep 'hipMemcpy' hip-trace.log
grep 'hipLaunchKernel' hip-trace.log

# Show timing information
grep -E '<[0-9]+\.[0-9]+>' hip-trace.log | head -20

# Count total calls
grep -c 'hip' hip-trace.log
```

## Requirements

- `ltrace` utility (install with `sudo apt-get install ltrace`)
- HIP-enabled application to trace
- Linux system with appropriate permissions

## Troubleshooting

- **"ltrace: command not found"**: Install ltrace with `sudo apt-get install ltrace`
- **Permission denied**: Ensure the application is executable
- **No HIP calls captured**: Check that the application actually uses HIP APIs
- **Large trace files**: Use `timeout` command to limit execution time for long-running applications
