# Makefile for AQL Packet Comparison Tools
#
# This Makefile builds the dumper programs and comparison tools
# needed to validate aql_c compatibility with aqlprofile_v2

# Compiler and flags
CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -O2 -g -std=c99
CXXFLAGS = -Wall -Wextra -O2 -g -std=c++17
LDFLAGS = -L./projects/aql_c_port/lib -L./projects/aqlprofile/lib

# Include directories
AQL_C_INCLUDES = -I./projects/aql_c_port/include
AQLPROFILE_INCLUDES = -I./projects/aqlprofile/src/core/include
HSA_INCLUDES = -I/opt/rocm/include

# Library dependencies
AQL_C_LIBS = -laql_c_port
AQLPROFILE_LIBS = -laqlprofile_v2 -lhsa-runtime64
COMMON_LIBS = -lm -lpthread

# Source files
AQL_C_DUMPER_SRC = aql_c_dumper.c
AQLPROFILE_V2_DUMPER_SRC = aqlprofile_v2_dumper.c

# Output binaries
AQL_C_DUMPER = aql_c_dumper
AQLPROFILE_V2_DUMPER = aqlprofile_v2_dumper

# Scripts
PACKET_COMPARE = packet_compare.py
TEST_RUNNER = run_comparison_tests.sh

# Default target
.PHONY: all clean help test install

all: $(AQL_C_DUMPER) $(AQLPROFILE_V2_DUMPER) scripts

# AQL_C dumper program
$(AQL_C_DUMPER): $(AQL_C_DUMPER_SRC)
	@echo "Building AQL_C dumper..."
	$(CC) $(CFLAGS) $(AQL_C_INCLUDES) -o $@ $< $(LDFLAGS) $(AQL_C_LIBS) $(COMMON_LIBS)
	@echo "AQL_C dumper built successfully"

# AQLProfile v2 dumper program
$(AQLPROFILE_V2_DUMPER): $(AQLPROFILE_V2_DUMPER_SRC)
	@echo "Building AQLProfile v2 dumper..."
	$(CC) $(CFLAGS) $(AQLPROFILE_INCLUDES) $(HSA_INCLUDES) -o $@ $< $(LDFLAGS) $(AQLPROFILE_LIBS) $(COMMON_LIBS)
	@echo "AQLProfile v2 dumper built successfully"

# Make scripts executable
scripts:
	@echo "Making scripts executable..."
	chmod +x $(PACKET_COMPARE)
	chmod +x $(TEST_RUNNER)
	@echo "Scripts made executable"

# Install target (copy to system locations)
install: all
	@echo "Installing comparison tools..."
	mkdir -p /usr/local/bin
	cp $(AQL_C_DUMPER) /usr/local/bin/
	cp $(AQLPROFILE_V2_DUMPER) /usr/local/bin/
	cp $(PACKET_COMPARE) /usr/local/bin/
	cp $(TEST_RUNNER) /usr/local/bin/
	@echo "Tools installed to /usr/local/bin"

# Run basic tests
test: all
	@echo "Running basic comparison tests..."
	./$(TEST_RUNNER) --test-suite basic --verbose

# Run full test suite
test-full: all
	@echo "Running full comparison test suite..."
	./$(TEST_RUNNER) --test-suite full --parallel 4 --verbose

# Run stress tests
test-stress: all
	@echo "Running stress tests..."
	./$(TEST_RUNNER) --test-suite stress --verbose

# Quick test for development
test-quick: all
	@echo "Running quick development test..."
	./$(TEST_RUNNER) --arch gfx942 --test-suite basic

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(AQL_C_DUMPER)
	rm -f $(AQLPROFILE_V2_DUMPER)
	rm -f *.o
	rm -rf test_results/
	rm -rf reference_results/
	@echo "Clean complete"

# Help target
help:
	@echo "AQL Packet Comparison Tools"
	@echo "==========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all dumper programs and prepare scripts"
	@echo "  clean        - Remove build artifacts and test results"
	@echo "  test         - Run basic comparison tests"
	@echo "  test-full    - Run comprehensive test suite"
	@echo "  test-stress  - Run stress tests with maximum configurations"
	@echo "  test-quick   - Run quick development test"
	@echo "  install      - Install tools to /usr/local/bin"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Dependencies:"
	@echo "  - aql_c_port library and headers"
	@echo "  - aqlprofile_v2 library and headers"
	@echo "  - HSA runtime (ROCm)"
	@echo "  - Python 3 for comparison scripts"
	@echo ""
	@echo "Usage examples:"
	@echo "  make all                 # Build everything"
	@echo "  make test                # Run basic tests"
	@echo "  make test-full           # Run comprehensive tests"
	@echo "  make clean               # Clean up"
	@echo ""
	@echo "Manual usage:"
	@echo "  ./aql_c_dumper --arch gfx942 --counter CPC:0:123 --output test.json"
	@echo "  ./aqlprofile_v2_dumper --arch gfx942 --counter CPC:0:123 --output ref.json"
	@echo "  ./packet_compare.py test.json ref.json --detailed"
	@echo "  ./run_comparison_tests.sh --arch gfx942 --test-suite basic --verbose"

# Dependencies and build order
$(AQL_C_DUMPER): | check-aql-c-deps
$(AQLPROFILE_V2_DUMPER): | check-aqlprofile-deps

.PHONY: check-aql-c-deps check-aqlprofile-deps

check-aql-c-deps:
	@echo "Checking AQL_C dependencies..."
	@if [ ! -d "./projects/aql_c_port/include" ]; then \
		echo "Error: AQL_C headers not found in ./projects/aql_c_port/include"; \
		echo "Please ensure aql_c_port project is built and headers are available"; \
		exit 1; \
	fi
	@echo "AQL_C dependencies OK"

check-aqlprofile-deps:
	@echo "Checking AQLProfile v2 dependencies..."
	@if [ ! -d "./projects/aqlprofile/src/core/include" ]; then \
		echo "Error: AQLProfile v2 headers not found in ./projects/aqlprofile/src/core/include"; \
		echo "Please ensure aqlprofile project is built and headers are available"; \
		exit 1; \
	fi
	@if [ ! -d "/opt/rocm/include" ]; then \
		echo "Warning: ROCm headers not found in /opt/rocm/include"; \
		echo "You may need to install ROCm or adjust HSA_INCLUDES path"; \
	fi
	@echo "AQLProfile v2 dependencies OK"

# Debug targets for development
debug-aql-c: $(AQL_C_DUMPER_SRC)
	$(CC) $(CFLAGS) -DDEBUG -g3 $(AQL_C_INCLUDES) -o $(AQL_C_DUMPER)_debug $< $(LDFLAGS) $(AQL_C_LIBS) $(COMMON_LIBS)

debug-aqlprofile: $(AQLPROFILE_V2_DUMPER_SRC)
	$(CC) $(CFLAGS) -DDEBUG -g3 $(AQLPROFILE_INCLUDES) $(HSA_INCLUDES) -o $(AQLPROFILE_V2_DUMPER)_debug $< $(LDFLAGS) $(AQLPROFILE_LIBS) $(COMMON_LIBS)

# Example test runs for different scenarios
example-single-counter:
	@echo "Example: Single counter test"
	./$(AQL_C_DUMPER) --arch gfx942 --counter CPC:0:123 --output example_aqlc.json --verbose
	./$(AQLPROFILE_V2_DUMPER) --arch gfx942 --counter CPC:0:123 --output example_aqlv2.json --verbose
	./$(PACKET_COMPARE) example_aqlc.json example_aqlv2.json --detailed

example-multi-counter:
	@echo "Example: Multi-counter test"
	./$(AQL_C_DUMPER) --arch gfx942 --counter CPC:0:123 --counter GRBM:0:456 --output multi_aqlc.json --verbose
	./$(AQLPROFILE_V2_DUMPER) --arch gfx942 --counter CPC:0:123 --counter GRBM:0:456 --output multi_aqlv2.json --verbose
	./$(PACKET_COMPARE) multi_aqlc.json multi_aqlv2.json --detailed

# Create test data for development
sample-data: all
	@echo "Generating sample test data..."
	mkdir -p samples
	./$(AQL_C_DUMPER) --arch gfx942 --counter CPC:0:123 --output samples/sample_aqlc.json
	./$(AQLPROFILE_V2_DUMPER) --arch gfx942 --counter CPC:0:123 --output samples/sample_aqlv2.json
	./$(AQL_C_DUMPER) --arch gfx942 --counter CPC:0:123 --format text --output samples/sample_aqlc.txt
	./$(AQLPROFILE_V2_DUMPER) --arch gfx942 --counter CPC:0:123 --format text --output samples/sample_aqlv2.txt
	@echo "Sample data generated in samples/ directory"

# Phony targets
.PHONY: example-single-counter example-multi-counter sample-data debug-aql-c debug-aqlprofile