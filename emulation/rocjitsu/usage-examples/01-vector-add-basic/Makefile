# Makefile for vector_add example
# Demonstrates basic HIP compilation and rocjitsu usage

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

# Source and target
SRC = $(SRC_DIR)/vector_add.cpp
TARGET = $(BUILD_DIR)/vector_add

.PHONY: all compile run run-verbose run-daemon clean help

# Default target (phony name "compile" — not "build", which conflicts with BUILD_DIR)
all: compile

# Build the example
compile: $(TARGET)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC) | $(BUILD_DIR)
	@echo "Compiling $(SRC) for $(ARCH)..."
	$(HIPCC) $(HIPFLAGS) $(CXXFLAGS) -o $@ $<
	@echo "Built: $@"

# Run with rocjitsu (basic)
run: $(TARGET)
	@echo "Running vector_add through rocjitsu..."
	@echo "Config: $(CONFIG)"
	@echo ""
	$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET)

# Run with verbose logging
run-verbose: $(TARGET)
	@echo "Running vector_add with verbose logging..."
	@echo "Config: $(CONFIG)"
	@echo ""
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET)

# Run in daemon mode
run-daemon: $(TARGET)
	@echo "Running vector_add in daemon mode..."
	@echo "Config: $(CONFIG)"
	@echo ""
	$(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(TARGET)

# Run with custom vector size
run-size: $(TARGET)
	@echo "Running vector_add with custom size..."
	@echo "Usage: make run-size SIZE=2048"
	$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET) $(SIZE)

# Run on different GPU architecture config
run-cdna3: $(TARGET)
	@echo "Running with CDNA3 (gfx942) configuration..."
	$(ROCJITSU_BIN) --config $(CONFIG_DIR)/amdgpu_cdna3_kmd.json -- $(TARGET)

# Clean build artifacts
clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR)
	@echo "Clean complete."

# Help target
help:
	@echo "Vector Add Example - Makefile Targets"
	@echo ""
	@echo "Targets:"
	@echo "  make               - Compile the example (default)"
	@echo "  make compile       - Same as make"
	@echo "  make run           - Run with rocjitsu (basic mode)"
	@echo "  make run-verbose   - Run with detailed logging (RJ_LOG=1)"
	@echo "  make run-daemon    - Run in daemon mode"
	@echo "  make run-size      - Run with custom size (make run-size SIZE=2048)"
	@echo "  make run-cdna3     - Run with CDNA3 GPU config"
	@echo "  make clean         - Remove build artifacts"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  HIPCC=$(HIPCC)"
	@echo "  ARCH=$(ARCH)"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  ROCJITSU_BIN=$(ROCJITSU_BIN)"
	@echo ""
	@echo "Examples:"
	@echo "  make ARCH=gfx942 build    - Build for CDNA3"
	@echo "  make SIZE=10000 run-size  - Run with 10000 elements"
