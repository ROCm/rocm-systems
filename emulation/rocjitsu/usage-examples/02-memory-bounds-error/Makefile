# Makefile for memory bounds error example

HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -Wall -O2
HIPFLAGS = --offload-arch=$(ARCH)

SRC_DIR = src
BUILD_DIR = build
ROCJITSU_ROOT = ../..
CONFIG ?= $(ROCJITSU_ROOT)/configs/amdgpu_cdna4_kmd.json
ROCJITSU_BIN ?= $(ROCJITSU_ROOT)/build/tools/rocjitsu/rocjitsu

TARGET_BUGGY = $(BUILD_DIR)/bounds_error
TARGET_FIXED = $(BUILD_DIR)/bounds_fixed

.PHONY: all compile run-buggy run-fixed compare clean help

all: compile

compile: $(TARGET_BUGGY) $(TARGET_FIXED)

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(TARGET_BUGGY): $(SRC_DIR)/bounds_error.cpp | $(BUILD_DIR)
	@echo "Building buggy version..."
	$(HIPCC) $(HIPFLAGS) $(CXXFLAGS) -o $@ $<

$(TARGET_FIXED): $(SRC_DIR)/bounds_fixed.cpp | $(BUILD_DIR)
	@echo "Building fixed version..."
	$(HIPCC) $(HIPFLAGS) $(CXXFLAGS) -o $@ $<

run-buggy: $(TARGET_BUGGY)
	@echo "Running buggy version..."
	$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_BUGGY) || true

run-fixed: $(TARGET_FIXED)
	@echo "Running fixed version..."
	$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_FIXED)

compare: $(TARGET_BUGGY) $(TARGET_FIXED)
	@echo "=== BUGGY VERSION ==="
	@$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_BUGGY) || true
	@echo ""
	@echo "=== FIXED VERSION ==="
	@$(ROCJITSU_BIN) --config $(CONFIG) -- $(TARGET_FIXED)

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Memory Bounds Error Example"
	@echo "  make build      - Build both versions"
	@echo "  make run-buggy  - Run buggy version"
	@echo "  make run-fixed  - Run fixed version"
	@echo "  make compare    - Compare both"
	@echo "  make clean      - Clean build"