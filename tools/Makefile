# Makefile for dat_reader and dat_generator

# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
ROCM_PATH := /opt/rocm

# Include paths
INCLUDES := -I$(ROCM_PATH)/include

# Library paths and libraries
LDFLAGS := -L$(ROCM_PATH)/lib
LIBS := -lrocprofiler-sdk

# Defines
DEFINES := -D__HIP_PLATFORM_AMD__=1

# Targets
READER := dat_reader
GENERATOR := dat_generator
TARGETS := $(READER) $(GENERATOR)

# Default target
all: $(TARGETS)

# Build dat_reader
$(READER): dat_reader.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< $(LDFLAGS) $(LIBS) -o $@

# Build dat_generator
$(GENERATOR): dat_generator.cpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $< $(LDFLAGS) $(LIBS) -o $@

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Install (optional)
install: $(TARGETS)
	install -m 755 $(READER) /usr/local/bin/
	install -m 755 $(GENERATOR) /usr/local/bin/

# Phony targets
.PHONY: all clean install
