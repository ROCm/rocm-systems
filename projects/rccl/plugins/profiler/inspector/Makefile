# Standalone Makefile for NCCL Inspector (independent from main NCCL)
#
# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
#
# Sources stay byte-identical to upstream and are hipified at build time.

# Compiler settings
ROCM_PATH ?= /opt/rocm
# ?= would lose to make's built-in CXX default (g++); only replace that default.
ifeq ($(origin CXX),default)
CXX := $(ROCM_PATH)/bin/amdclang++
endif
HIPIFY_PL_EXE ?= $(ROCM_PATH)/bin/hipify-perl

# Build options
DEBUG ?= 0
ASAN ?= 0
UBSAN ?= 0
NCCL_INSPECTOR_ENABLE_WARN ?= 0

CXXSTD ?= -std=c++17

# Build directories
OBJDIR := obj
SRCDIR := .
HIPIFY_DIR := hipify-inspector

# Compiler flags
CXXFLAGS := $(CXXSTD) -fPIC -Wall -Wextra -O2 -D__HIP_PLATFORM_AMD__ -DNCCL_INSPECTOR_ENABLE_WARN=$(NCCL_INSPECTOR_ENABLE_WARN)
INCLUDES := -I$(HIPIFY_DIR) -I$(HIPIFY_DIR)/nccl -I$(ROCM_PATH)/include
LDFLAGS := -shared
LIBS := -ldl -lpthread

# Debug flags
ifneq ($(DEBUG), 0)
CXXFLAGS += -g -DDEBUG
endif

# Sanitizer flags
ifneq ($(ASAN), 0)
CXXFLAGS += -fsanitize=address
LDFLAGS += -fsanitize=address -shared-libasan
endif

ifneq ($(UBSAN), 0)
CXXFLAGS += -fsanitize=undefined
LDFLAGS += -fsanitize=undefined
endif

# Source files
SRCS := inspector.cc \
	inspector_json.cc \
	inspector_prom.cc \
	inspector_plugin.cc \
	inspector_cudawrap.cc \
	inspector_ring.cc \
	inspector_event_pool.cc \
	json.cc \
	version.cc

OBJS := $(SRCS:%.cc=$(OBJDIR)/%.o)

HIPIFY_STAMP := $(HIPIFY_DIR)/.hipify.stamp

# Target library
TARGET := librccl-profiler-inspector.so

# Build rules
.PHONY: all clean help

# Without this make treats the hipified copies as intermediates and re-does them each build.
.SECONDARY: $(addprefix $(HIPIFY_DIR)/,$(SRCS))

all: $(TARGET)

$(TARGET): $(OBJDIR) $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)
	@echo "Built standalone RCCL Inspector: $@"


# Fix what hipify cannot reach: INSPECTOR_CUPFN pastes pfn_##sym, and the dlopen'd
# library is a string literal.
$(HIPIFY_STAMP): $(wildcard $(SRCDIR)/*.cc) $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/nccl/*.h)
	@mkdir -p $(HIPIFY_DIR)/nccl
	@cp -f $(SRCDIR)/*.cc $(SRCDIR)/*.h $(HIPIFY_DIR)/
	@cp -f $(SRCDIR)/nccl/*.h $(HIPIFY_DIR)/nccl/
	@rm -f $(HIPIFY_DIR)/*.prehip $(HIPIFY_DIR)/nccl/*.prehip
	@$(HIPIFY_PL_EXE) -inplace -quiet-warnings $(HIPIFY_DIR)/*.cc $(HIPIFY_DIR)/nccl/*.h \
	  `ls $(HIPIFY_DIR)/*.h | grep -v inspector_gpu_compat.h`
	@sed -i 's/\bpfn_cu/pfn_hip/g; s/\bPFN_cu/PFN_hip/g' $(HIPIFY_DIR)/*.cc $(HIPIFY_DIR)/*.h
	@sed -i 's/"libcuda\.so\.1"/"libamdhip64.so.6"/g; s/"libcuda\.so"/"libamdhip64.so"/g' $(HIPIFY_DIR)/*.cc
	@touch $@

$(HIPIFY_DIR)/%.cc: $(HIPIFY_STAMP)
	@true

$(HIPIFY_DIR)/version.cc: | $(HIPIFY_STAMP)
	@mkdir -p $(HIPIFY_DIR)
	@echo "// Auto-generated version info" > $@
	@echo "#ifdef __cplusplus" >> $@
	@echo "extern \"C\" {" >> $@
	@echo "#endif" >> $@
	@echo "const char* get_git_version_info() {" >> $@
	@if command -v git >/dev/null 2>&1; then \
		echo "  return \"$(shell git describe --always --dirty 2>/dev/null || echo 'unknown')\"; " >> $@; \
	else \
		echo "  return \"standalone-build\"; " >> $@; \
	fi
	@echo "}" >> $@
	@echo "#ifdef __cplusplus" >> $@
	@echo "}" >> $@
	@echo "#endif" >> $@

$(OBJDIR)/%.o: $(HIPIFY_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET) $(HIPIFY_DIR)

# Help target
help:
	@echo "Standalone RCCL Inspector Build System"
	@echo ""
	@echo "Targets:"
	@echo "  all     - Build the inspector library (default)"
	@echo "  clean   - Clean build artifacts"
	@echo "  help    - Show this help"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1                      - Enable debug build"
	@echo "  ASAN=1                       - Enable address sanitizer"
	@echo "  UBSAN=1                      - Enable undefined behavior sanitizer"
	@echo "  NCCL_INSPECTOR_ENABLE_WARN=1 - Emit inspector messages at WARN level (default: INFO)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  ROCM_PATH - Path to ROCm installation (default: /opt/rocm)"
	@echo "  CXX       - C++ compiler (default: \$$(ROCM_PATH)/bin/amdclang++)"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make DEBUG=1"
	@echo "  make ROCM_PATH=/opt/rocm"
