#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# RCCL Examples Main Makefile (ported from NCCL examples)

# Define all category directories. 06_device_api is RCCL-specific and lives
# alongside but uses its own build system, so it's intentionally omitted here.
CATEGORIES := 01_communicators 02_point_to_point 03_collectives 04_user_buffer_registration 05_symmetric_memory

# Default target
all: $(CATEGORIES)

# Build all categories
$(CATEGORIES):
	@echo "Building $@..."
	@$(MAKE) -C $@

# Clean all categories
clean:
	@echo "Cleaning all examples..."
	@for category in $(CATEGORIES); do \
		$(MAKE) -C $$category clean; \
	done

# Test all examples
test: all
	@echo "Testing all examples..."
	@for category in $(CATEGORIES); do \
		$(MAKE) -C $$category test; \
	done

# Install all examples
install: all
	@echo "Installing all examples..."
	@for category in $(CATEGORIES); do \
		$(MAKE) -C $$category install; \
	done

# Help target
help:
	@echo "RCCL Examples Main Makefile"
	@echo "==========================="
	@echo ""
	@echo "Available targets:"
	@echo "  all      - Build all examples"
	@echo "  clean    - Clean all build artifacts"
	@echo "  test     - Build and test all examples"
	@echo "  install  - Install all examples"
	@echo "  help     - Show this help message"
	@echo ""

.PHONY: all clean test install help $(CATEGORIES)
