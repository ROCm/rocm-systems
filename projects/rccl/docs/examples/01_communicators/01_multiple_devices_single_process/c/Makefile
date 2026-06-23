#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Include common build rules
<<<<<<< HEAD:projects/rccl/examples/01_communicators/01_multiple_devices_single_process/Makefile
include ../../makefiles/rccl_examples.mk
=======
include ../../../../../makefiles/common.mk
include ../../../../../makefiles/examples.mk
>>>>>>> v2.30.7-1:projects/rccl/docs/examples/01_communicators/01_multiple_devices_single_process/c/Makefile

# Target executable
TARGET = multiple_devices_single_process

# Source files
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

# Compile source files
%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: $(TARGET)
	@echo "Testing $(TARGET)..."
	@echo "Running with all available GPUs"
	./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "NCCL Example (C): Multiple Devices Single Process"
	@echo "================================================="
	@echo ""
	@echo "This example shows how to use ncclCommInitAll to create"
	@echo "communicators for multiple GPUs in a single process."
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build the example (default)"
	@echo "  test      - Build and run test with all GPUs"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help      - Show this help"

.PHONY: all test clean install help
