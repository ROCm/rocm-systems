HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -O2
BUILD_DIR = build
ROCM_PATH ?= /opt/rocm
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG ?= ../../configs/amdgpu_cdna4_kmd.json

# rocBLAS lives in the ROCm install.
ROCBLAS_FLAGS = -I$(ROCM_PATH)/include -L$(ROCM_PATH)/lib -lrocblas -Wl,-rpath,$(ROCM_PATH)/lib

.PHONY: all run run-log clean help

all: $(BUILD_DIR)/gemm_test

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/gemm_test: src/gemm_test.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -o $@ $< $(ROCBLAS_FLAGS)

# Buggy vs fixed call + host golden verification.
run: $(BUILD_DIR)/gemm_test
	$(ROCJITSU_BIN) --config $(CONFIG) -- $<

# Same, with dispatch logging: shows the rocBLAS GEMM dispatch and, crucially,
# "[rocjitsu] mfma detected in dispatch N" — confirming matrix-core (MFMA) use.
run-log: $(BUILD_DIR)/gemm_test
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $<

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "rocBLAS GEMM debugging example - targets:"
	@echo "  make          - build (links rocBLAS)"
	@echo "  make run      - buggy (row-major) vs fixed (column-major) sgemm + host check"
	@echo "  make run-log  - same with RJ_LOG=1 (GEMM dispatch + 'mfma detected')"
	@echo "  make clean    - remove build/"
	@echo ""
	@echo "rocBLAS is column-major; the host golden check catches the classic"
	@echo "row-major misuse. rocjitsu confirms matrix-core (MFMA) usage but does"
	@echo "not verify numerics."
