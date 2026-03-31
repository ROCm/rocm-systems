# Makefile for amdsmi_wrap standalone test
ROCM_PATH ?= /opt/rocm
CXX := $(shell command -v hipcc 2>/dev/null || echo g++)

RCCL_ROOT := ../..
RCCL_SRC := $(RCCL_ROOT)/src
HIPIFY_DIR := ./hipify

INCLUDES := -I. -I$(HIPIFY_DIR) -I$(RCCL_SRC)/include -I$(RCCL_SRC) -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/rocm_smi -I$(ROCM_PATH)/include/amd_smi
CXXFLAGS := -std=c++17 -Wall -Wextra -Wno-unused-parameter -D__HIP_PLATFORM_AMD__ -DRCCL_STANDALONE_TEST $(INCLUDES)
LDFLAGS := -L$(ROCM_PATH)/lib -L$(ROCM_PATH)/lib64
LIBS := -lrocm_smi64 -lpthread -ldl -lamdhip64

TARGET := amdsmi_wrap_test
OBJS := amdsmi_wrap_test.o amdsmi_wrap.o alt_rsmi.o test_utils.o

all: core.h $(TARGET)

core.h: test_stubs.h
	ln -sf test_stubs.h core.h
	ln -sf test_stubs.h utils.h

$(HIPIFY_DIR):
	mkdir -p $(HIPIFY_DIR)

$(HIPIFY_DIR)/amdsmi_wrap.cc: $(RCCL_SRC)/misc/amdsmi_wrap.cc | $(HIPIFY_DIR)
	@echo "Hipifying amdsmi_wrap.cc..."
	hipify-perl -quiet-warnings $< -o $@

$(HIPIFY_DIR)/alt_rsmi.cc: $(RCCL_SRC)/misc/alt_rsmi.cc | $(HIPIFY_DIR)
	@echo "Hipifying alt_rsmi.cc..."
	hipify-perl -quiet-warnings $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

amdsmi_wrap_test.o: amdsmi_wrap_test.cpp test_stubs.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

amdsmi_wrap.o: $(HIPIFY_DIR)/amdsmi_wrap.cc core.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

alt_rsmi.o: $(HIPIFY_DIR)/alt_rsmi.cc
	$(CXX) $(CXXFLAGS) -DARSMI_TEST_BUILD -c -o $@ $<

test_utils.o: test_utils.cc test_stubs.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

run-amdsmi: $(TARGET)
	RCCL_USE_AMD_SMI_LIB=1 ./$(TARGET)

clean:
	rm -f $(TARGET) *.o core.h utils.h
	rm -rf $(HIPIFY_DIR)

info:
	@echo "Compiler: $(CXX)"

.PHONY: all clean run run-amdsmi info
