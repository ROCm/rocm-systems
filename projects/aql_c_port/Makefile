# Makefile for AQL C Port Library
#
# This Makefile builds the AQL packet generation library and examples
# for both userspace and kernel module use.

# Project configuration
PROJECT_NAME = aql_c_port
VERSION = 1.0.0

# Directories
SRC_DIR = src
INC_DIR = include
EXAMPLES_DIR = examples
TESTS_DIR = tests
BUILD_DIR = build
DIST_DIR = dist

# Compiler configuration
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -Werror -I$(INC_DIR)
CFLAGS += -fPIC -O2 -g

# Debug flags
ifdef DEBUG
	CFLAGS += -DAQL_DEBUG_TRACE -DDEBUG -O0
endif

# Kernel build configuration
KERNEL_DIR ?= /lib/modules/$(shell uname -r)/build
KBUILD_CFLAGS += -I$(PWD)/$(INC_DIR)

# Source files
LIB_SOURCES = $(SRC_DIR)/aql_cmd_buffer.c \
              $(SRC_DIR)/aql_packet.c \
              $(SRC_DIR)/aql_arch_detect.c \
              $(SRC_DIR)/aql_gfx9_ops.c \
              $(SRC_DIR)/aql_gfx12_ops.c \
              $(SRC_DIR)/aql_pmc_interface.c

LIB_HEADERS = $(INC_DIR)/aql_types.h \
              $(INC_DIR)/aql_arch_ops.h \
              $(INC_DIR)/aql_cmd_buffer.h \
              $(INC_DIR)/aql_gfx9_defs.h \
              $(INC_DIR)/aql_gfx12_defs.h \
              $(INC_DIR)/aql_pmc_interface.h

EXAMPLE_SOURCES = $(EXAMPLES_DIR)/simple_example.c

# Object files
LIB_OBJECTS = $(LIB_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
EXAMPLE_OBJECTS = $(EXAMPLE_SOURCES:$(EXAMPLES_DIR)/%.c=$(BUILD_DIR)/examples/%.o)

# Target files
STATIC_LIB = $(BUILD_DIR)/lib$(PROJECT_NAME).a
SHARED_LIB = $(BUILD_DIR)/lib$(PROJECT_NAME).so
EXAMPLE_BIN = $(BUILD_DIR)/simple_example

# Phony targets
.PHONY: all clean userspace kernel examples install uninstall dist help

# Default target
all: userspace

# Help target
help:
	@echo "AQL C Port Library Build System"
	@echo "==============================="
	@echo ""
	@echo "Targets:"
	@echo "  all         - Build userspace library (default)"
	@echo "  userspace   - Build userspace static and shared libraries"
	@echo "  examples    - Build example programs"
	@echo "  kernel      - Build kernel module (future)"
	@echo "  clean       - Clean build artifacts"
	@echo "  install     - Install library and headers"
	@echo "  uninstall   - Remove installed files"
	@echo "  dist        - Create distribution package"
	@echo "  help        - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1     - Build with debug symbols and tracing"
	@echo "  PREFIX=path - Installation prefix (default: /usr/local)"
	@echo ""

# Userspace build targets
userspace: $(STATIC_LIB) $(SHARED_LIB)

# Static library
$(STATIC_LIB): $(LIB_OBJECTS) | $(BUILD_DIR)
	@echo "Creating static library: $@"
	ar rcs $@ $^

# Shared library
$(SHARED_LIB): $(LIB_OBJECTS) | $(BUILD_DIR)
	@echo "Creating shared library: $@"
	$(CC) -shared -o $@ $^

# Object file compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c $(LIB_HEADERS) | $(BUILD_DIR)
	@echo "Compiling: $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Example programs
examples: $(EXAMPLE_BIN)

$(BUILD_DIR)/examples/%.o: $(EXAMPLES_DIR)/%.c $(LIB_HEADERS) | $(BUILD_DIR)/examples
	@echo "Compiling example: $<"
	$(CC) $(CFLAGS) -c $< -o $@

$(EXAMPLE_BIN): $(EXAMPLE_OBJECTS) $(STATIC_LIB)
	@echo "Linking example: $@"
	$(CC) $(CFLAGS) $^ -o $@

# Directory creation
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/examples:
	mkdir -p $(BUILD_DIR)/examples

# Kernel module target (placeholder for future implementation)
kernel:
	@echo "Kernel module build not yet implemented"
	@echo "This will build a kernel module using the same source code"
	@echo "with kernel-specific compilation flags."

# Installation
PREFIX ?= /usr/local
INSTALL_LIB_DIR = $(PREFIX)/lib
INSTALL_INC_DIR = $(PREFIX)/include/aql

install: userspace
	@echo "Installing AQL C library to $(PREFIX)"
	mkdir -p $(INSTALL_LIB_DIR)
	mkdir -p $(INSTALL_INC_DIR)
	cp $(STATIC_LIB) $(INSTALL_LIB_DIR)/
	cp $(SHARED_LIB) $(INSTALL_LIB_DIR)/
	cp $(LIB_HEADERS) $(INSTALL_INC_DIR)/
	ldconfig

uninstall:
	@echo "Uninstalling AQL C library from $(PREFIX)"
	rm -f $(INSTALL_LIB_DIR)/lib$(PROJECT_NAME).a
	rm -f $(INSTALL_LIB_DIR)/lib$(PROJECT_NAME).so
	rm -rf $(INSTALL_INC_DIR)
	ldconfig

# Distribution package
dist: clean
	@echo "Creating distribution package"
	mkdir -p $(DIST_DIR)
	tar -czf $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION).tar.gz \
		--exclude=$(DIST_DIR) \
		--exclude=$(BUILD_DIR) \
		--exclude=.git \
		.

# Clean targets
clean:
	@echo "Cleaning build artifacts"
	rm -rf $(BUILD_DIR)
	rm -rf $(DIST_DIR)

# Test targets (placeholder for future implementation)
test: userspace
	@echo "Running basic validation"
	@if [ -f $(EXAMPLE_BIN) ]; then \
		echo "Running simple example..."; \
		$(EXAMPLE_BIN); \
	else \
		echo "Example not built. Run 'make examples' first."; \
	fi

# Kernel module Makefile (for out-of-tree builds)
# This would be used when building as a kernel module
ifneq ($(KERNELRELEASE),)
	# We're being called from kernel build system
	obj-m := aql_c_port_km.o
	aql_c_port_km-objs := $(LIB_SOURCES:.c=.o)
	ccflags-y += -I$(src)/$(INC_DIR) -D__KERNEL__
else
	# Normal Makefile execution
endif

# Development targets
format:
	@echo "Formatting source code (if clang-format available)"
	@if command -v clang-format >/dev/null 2>&1; then \
		find $(SRC_DIR) $(INC_DIR) $(EXAMPLES_DIR) -name "*.c" -o -name "*.h" | \
		xargs clang-format -i; \
		echo "Code formatted"; \
	else \
		echo "clang-format not found, skipping formatting"; \
	fi

check:
	@echo "Running static analysis (if available)"
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --enable=all --std=c99 -I$(INC_DIR) $(SRC_DIR); \
	else \
		echo "cppcheck not found, skipping static analysis"; \
	fi

# Debug build
debug:
	$(MAKE) DEBUG=1 all examples

# Show build information
info:
	@echo "Build Configuration:"
	@echo "  Project: $(PROJECT_NAME) v$(VERSION)"
	@echo "  CC: $(CC)"
	@echo "  CFLAGS: $(CFLAGS)"
	@echo "  Source files: $(words $(LIB_SOURCES))"
	@echo "  Header files: $(words $(LIB_HEADERS))"
	@echo "  Build directory: $(BUILD_DIR)"
	@echo "  Install prefix: $(PREFIX)"