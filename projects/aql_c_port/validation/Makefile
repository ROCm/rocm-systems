# Validation Tools Makefile for AQL C Library
# Builds validation and testing tools for counter block validation

CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -g -O2
INCLUDES = -I../include -I/opt/rocm/include
LDFLAGS = -L/opt/rocm/lib -L../build
LIBS = -laql_c_port -lhsa-runtime64

# Build directory
BUILD_DIR = build
LIB_DIR = ../build

# Source files
SOURCES = aql_c_pmc_dumper.c
TARGETS = $(BUILD_DIR)/aql_c_pmc_dumper

# Test data files
TEST_FILES = test_all_gfx9_blocks.csv test_all_gfx12_blocks.csv test_counters_gfx942.csv

.PHONY: all clean test test-gfx9 test-gfx12 help

all: $(BUILD_DIR) $(TARGETS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build the PMC dumper tool
$(BUILD_DIR)/aql_c_pmc_dumper: aql_c_pmc_dumper.c | $(BUILD_DIR)
	@echo "Building AQL C PMC Dumper..."
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS) $(LIBS)

# Ensure the main library is built
$(LIB_DIR)/libaql_c_port.a:
	@echo "Building AQL C library..."
	cd .. && $(MAKE)

# Run comprehensive validation tests
test: test-gfx9 test-gfx12

test-gfx9: $(BUILD_DIR)/aql_c_pmc_dumper $(LIB_DIR)/libaql_c_port.a
	@echo "Running GFX9 validation tests..."
	./$(BUILD_DIR)/aql_c_pmc_dumper test_all_gfx9_blocks.csv > $(BUILD_DIR)/gfx9_validation_results.txt 2>&1
	@echo "GFX9 validation results saved to $(BUILD_DIR)/gfx9_validation_results.txt"

test-gfx12: $(BUILD_DIR)/aql_c_pmc_dumper $(LIB_DIR)/libaql_c_port.a
	@echo "Running GFX12 validation tests..."
	./$(BUILD_DIR)/aql_c_pmc_dumper test_all_gfx12_blocks.csv > $(BUILD_DIR)/gfx12_validation_results.txt 2>&1
	@echo "GFX12 validation results saved to $(BUILD_DIR)/gfx12_validation_results.txt"

test-basic: $(BUILD_DIR)/aql_c_pmc_dumper $(LIB_DIR)/libaql_c_port.a
	@echo "Running basic counter validation test..."
	./$(BUILD_DIR)/aql_c_pmc_dumper test_counters_gfx942.csv > $(BUILD_DIR)/basic_validation_results.txt 2>&1
	@echo "Basic validation results saved to $(BUILD_DIR)/basic_validation_results.txt"

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Display help
help:
	@echo "AQL C Library Validation Tools"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build all validation tools"
	@echo "  test         - Run all validation tests (GFX9 + GFX12)"
	@echo "  test-gfx9    - Run GFX9-specific validation tests"
	@echo "  test-gfx12   - Run GFX12-specific validation tests"
	@echo "  test-basic   - Run basic counter validation test"
	@echo "  clean        - Remove build artifacts"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - ROCm installed in /opt/rocm"
	@echo "  - AQL C library built (../build/libaql_c_port.a)"
	@echo ""
	@echo "Test files:"
	@echo "  - test_all_gfx9_blocks.csv   - Comprehensive GFX9 counter test"
	@echo "  - test_all_gfx12_blocks.csv  - Comprehensive GFX12 counter test"
	@echo "  - test_counters_gfx942.csv   - Basic counter validation test"