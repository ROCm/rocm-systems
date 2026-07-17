# Makefile for data race detection example
# Demonstrates race detection and atomic operations

# Compiler and flags
HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -Wall -O2
HIPFLAGS = --offload-arch=$(ARCH)

# Paths
SRC_DIR = src
BUILD_DIR = build
ROCJITSU_ROOT = ../..
CONFIG_DIR = $(ROCJITSU_ROOT)/configs
ROCJITSU_BIN ?= $(ROCJITSU_ROOT)/build/tools/rocjitsu/rocjitsu

# Configuration
CONFIG ?= $(CONFIG_DIR)/amdgpu_cdna4_kmd.json

# Targets
TARGET_RACE = $(BUILD_DIR)/histogram_race
TARGET_FIXED = $(BUILD_DIR)/histogram_fixed

# Sources
SRC_RACE = $(SRC_DIR)/histogram_race.cpp
SRC_FIXED = $(SRC_DIR)/histogram_fixed.cpp

.PHONY: all compile run-race run-fixed compare clean help

# Default: build both versions
all: compile

# Build both versions
compile: $(TARGET_RACE) $(TARGET_FIXED)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# Build buggy version
$(TARGET_RACE): $(SRC_RACE) | $(BUILD_DIR)
	@echo "Compiling histogram_race.cpp (with data race)..."
	$(HIPCC) $(HIPFLAGS) $(CXXFLAGS) -o $@ $<
	@echo "Built: $@"

# Build fixed version
$(TARGET_FIXED): $(SRC_FIXED) | $(BUILD_DIR)
	@echo "Compiling histogram_fixed.cpp (with __syncthreads fix)..."
	$(HIPCC) $(HIPFLAGS) $(CXXFLAGS) -o $@ $<
	@echo "Built: $@"

# Run buggy version with race detector
run-race: $(TARGET_RACE)
	@echo "========================================="
	@echo "Running BUGGY version (with race detector)"
	@echo "========================================="
	@echo ""
	RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_RACE) || true
	@echo ""

# Run fixed version with race detector
run-fixed: $(TARGET_FIXED)
	@echo "========================================="
	@echo "Running FIXED version (with race detector)"
	@echo "========================================="
	@echo ""
	RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_FIXED)
	@echo ""

# Run buggy version WITHOUT race detector (shows incorrect results)
run-race-no-detector: $(TARGET_RACE)
	@echo "Running buggy version WITHOUT race detector..."
	@echo "(Results will be incorrect, but no race warnings)"
	@echo ""
	$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_RACE) || true

# Compare both versions
compare: $(TARGET_RACE) $(TARGET_FIXED)
	@echo "========================================="
	@echo "COMPARISON: Buggy vs Fixed"
	@echo "========================================="
	@echo ""
	@echo "--- Running BUGGY version ---"
	@RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_RACE) 2>&1 | head -30 || true
	@echo ""
	@echo "========================================="
	@echo ""
	@echo "--- Running FIXED version ---"
	@RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_FIXED) 2>&1 | head -30
	@echo ""
	@echo "========================================="
	@echo "SUMMARY:"
	@echo "  Buggy version: Has races, incorrect results"
	@echo "  Fixed version: No races, correct results"
	@echo "========================================="

# Run with verbose logging
run-verbose: $(TARGET_RACE)
	@echo "Running with verbose logging and race detection..."
	RJ_LOG=1 RJ_RACE=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_RACE) || true

# Run with output to file
run-to-file: $(TARGET_RACE)
	@mkdir -p logs
	@echo "Running with file output (logs/race_report.txt)..."
	RJ_RACE=1 RJ_SINKS=file RJ_SINK_DIR=logs $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_RACE) || true
	@echo ""
	@echo "Race report saved to: logs/"
	@ls -lh logs/

# Clean build artifacts
clean:
	@echo "Cleaning build directory and logs..."
	rm -rf $(BUILD_DIR) logs
	@echo "Clean complete."

# Help
help:
	@echo "Data Race Detection Example - Makefile Targets"
	@echo ""
	@echo "Targets:"
	@echo "  make build              - Build both versions (buggy + fixed)"
	@echo "  make run-race           - Run buggy version with race detector"
	@echo "  make run-fixed          - Run fixed version with race detector"
	@echo "  make compare            - Compare both versions side-by-side"
	@echo "  make run-verbose        - Run with verbose logging"
	@echo "  make run-to-file        - Save race reports to logs/"
	@echo "  make clean              - Remove build artifacts"
	@echo "  make help               - Show this help"
	@echo ""
	@echo "Key Environment Variables:"
	@echo "  RJ_RACE=1               - Enable race detection"
	@echo "  RJ_LOG=1                - Verbose logging"
	@echo "  RJ_SINK_DIR=<dir>       - Output directory for logs"
	@echo ""
	@echo "Examples:"
	@echo "  make compare            - See buggy vs fixed side-by-side"
	@echo "  make run-to-file        - Save detailed race reports"