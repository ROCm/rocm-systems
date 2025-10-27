# rdhc
Rocm Deployment Health Check Tool


## Features of the ROCm Deployment Health Check Tool

1. **Cross-Platform Support**: Works on Ubuntu, RHEL, and SLES distributions
2. **Comprehensive Testing**:
   - Default tests (GPU presence, driver status, rocminfo, rocm-smi)
   - Library dependency verification
   - Check some kernel parameters and ENV variables presence
   - Component-specific tests
     - Build and test the test program available from rocm-examples git repo dynamically.
3. **Dynamic Component Detection**: Identifies installed ROCm components using distribution-specific package manager commands
4. **Flexible Reporting**:
   - Pretty table output for terminal display
   - JSON export for further analysis or integration
5. **Configurable Verbosity**: Through command-line options (`-v` for verbose, `-s` for silent)

## Install dependency pip packages

```bash
sudo pip3 install -r requirements.txt

```
## Usage

```bash
./rdhc.py -h
usage: sudo -E rdhc.py [options]

ROCm Deployment Health Check Tool

optional arguments:
  -h, --help            show this help message and exit
  --quick               Run quick tests only (default)
  --all                 Default tests + Compile and executes simple program for each component.
  -v, --verbose         Enable verbose output
  -s, --silent          Silent mode (errors only)
  -j FILE, --json FILE  Export results to JSON file
  -d DIR, --dir DIR     Directory path for temporary files (default: /tmp/rdhc/)

Usage examples:
# Run quick test (default tests only)
sudo -E ./rdhc.py

# Run all tests including compile and execute the rocm-example program for each component
sudo -E ./rdhc.py --all

# Run all tests with verbose output
sudo -E ./rdhc.py --all -v

# Enable verbose output
sudo -E ./rdhc.py -v

# Run in silent mode (only errors shown)
sudo -E ./rdhc.py -s

# Export results to a specific JSON file
sudo -E ./rdhc.py --all --json rdhc-results.json

# Specify a directory for temp files and logs (default: /tmp/rdhc/)
sudo -E ./rdhc.py -d /home/user/rdhc-dir/

```
## RDHC Environment VARIABLES
RDHC tool will use the following ENV varaibles and act accordingly if they are set.
```bash
# ROCm installation path can be set by the below ENV varaible. Default is "/opt/rocm/"
export ROCM_PATH="/opt/rocm"

# For library dependency validation, the lib search depth can be set by the below ENV.
# Default is full depth. It checks for all the lib files in ROCM_PATH/lib/ folder recursively.
export LIBDIR_MAX_DEPTH=""

# if you want to check the libs only from the ROCM_PATH/lib/ folder set the depth as 1.
export LIBDIR_MAX_DEPTH=1

```
The tool is designed to be easily extended with additional component tests by
adding new test methods following the naming convention `test_check_component_name()`.
