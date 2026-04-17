# Convenience Makefile for smoke_test_compare_transpilers.
#
# The tool links against libhsa-runtime64.so from a Salmon-enabled ROCR build.
# It does not integrate with the transpiler's main CMake project on purpose —
# that project targets the transpiler library, while this tool is a harness
# around the already-built ROCR runtime.
#
# Usage:
#   make ROCR_BUILD=/path/to/rocr/build           # release build
#   make ROCR_BUILD=/path/to/rocr/build debug     # with -O0 -g
#   make clean
#
# Required:
#   ROCR_BUILD                    root of a Salmon-enabled ROCR build, such
#                                 that $(ROCR_BUILD)/rocr/include/hsa/hsa.h
#                                 and $(ROCR_BUILD)/rocr/lib/libhsa-runtime64.so
#                                 both exist.

ROCR_BUILD ?=
ifeq ($(strip $(ROCR_BUILD)),)
$(error ROCR_BUILD is not set. Point it at a Salmon-enabled ROCR build root, \
e.g. make ROCR_BUILD=$$HOME/rocr-runtime/build)
endif

ROCR_INCLUDE := $(ROCR_BUILD)/rocr/include
ROCR_LIBDIR  := $(ROCR_BUILD)/rocr/lib

ifeq ($(wildcard $(ROCR_INCLUDE)/hsa/hsa.h),)
$(error cannot find $(ROCR_INCLUDE)/hsa/hsa.h; is ROCR_BUILD correct?)
endif
ifeq ($(wildcard $(ROCR_LIBDIR)/libhsa-runtime64.so),)
$(error cannot find $(ROCR_LIBDIR)/libhsa-runtime64.so; is ROCR_BUILD correct?)
endif

CXX       ?= g++
CXXSTD    := -std=c++17
WARN      := -Wall -Wextra -Wno-unused-parameter
OPT       := -O2
CXXFLAGS  += $(CXXSTD) $(WARN) $(OPT) -I$(ROCR_INCLUDE)
LDFLAGS   += -L$(ROCR_LIBDIR) -Wl,-rpath,$(ROCR_LIBDIR)
LDLIBS    += -lhsa-runtime64

BIN := smoke_test_compare_transpilers
SRC := smoke_test_compare_transpilers.cpp

.PHONY: all debug clean
all: $(BIN)

debug: OPT := -O0 -g
debug: $(BIN)

$(BIN): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(BIN)
