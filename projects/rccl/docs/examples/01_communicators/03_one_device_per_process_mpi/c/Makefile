#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# This examples needs to be built with MPI support
MPI = 1

# Include common build rules
<<<<<<< HEAD:projects/rccl/examples/01_communicators/03_one_device_per_process_mpi/Makefile
include ../../makefiles/rccl_examples.mk
=======
include ../../../../../makefiles/common.mk
include ../../../../../makefiles/examples.mk
>>>>>>> v2.30.7-1:projects/rccl/docs/examples/01_communicators/03_one_device_per_process_mpi/c/Makefile

# Target executable
TARGET = one_device_per_process_mpi

# Source files
SOURCES = main.cc
OBJECTS = $(SOURCES:.cc=.o)

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(OBJECTS)
	$(MPICXX) $(CXXFLAGS) $(OBJECTS) $(LIBRARIES) $(LDFLAGS) -o $@
	@echo "Built target $@"

# Compile source files
%.o: %.cc
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Test target
test: $(TARGET)
	@echo "Testing $(TARGET)..."
	@echo "Running with 1 process"
	$(MPIRUN) -np 1 ./$(TARGET)
	@echo ""
	@echo "Running with 2 processes"
	$(MPIRUN) -np 2 ./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target
install: $(TARGET)
	@mkdir -p $(PREFIX)/bin
	cp $(TARGET) $(PREFIX)/bin/

# Help
help:
	@echo "NCCL Example (C): One Device per Process (MPI)"
	@echo "=============================================="
	@echo ""
	@echo "This example shows how to use ncclCommInitRank to create"
	@echo "communicators for multiple GPUs using MPI."
	@echo ""
	@echo "Targets:"
	@echo "  all           - Build the example (default)"
	@echo "  test          - Build and run tests with different process counts"
	@echo "  clean         - Remove build artifacts"
	@echo "  install       - Install to PREFIX/bin (default: /usr/local/bin)"
	@echo "  help          - Show this help"

.PHONY: all test clean install help
