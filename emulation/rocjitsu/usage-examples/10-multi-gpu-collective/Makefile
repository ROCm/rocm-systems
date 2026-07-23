HIPCC ?= hipcc
ARCH ?= gfx950
CXXFLAGS = -std=c++20 -O2
BUILD_DIR = build
ROCJITSU_BIN ?= ../../build/tools/rocjitsu/rocjitsu
CONFIG_2GPU ?= ../../configs/amdgpu_cdna4_kmd_2gpu.json

.PHONY: all run-2gpu run-2gpu-log clean help

all: $(BUILD_DIR)/multi_gpu

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/multi_gpu: src/multi_gpu.cpp | $(BUILD_DIR)
	$(HIPCC) --offload-arch=$(ARCH) $(CXXFLAGS) -o $@ $<

# Hand-written all-reduce across the 2 emulated GPUs (daemon mode for PyTorch-like
# multi-device apps; a plain HIP app works either way).
run-2gpu: $(BUILD_DIR)/multi_gpu
	$(ROCJITSU_BIN) --daemon --config $(CONFIG_2GPU) -- $<

# Local mode + RJ_LOG=1 to see the per-GPU kernel dispatches (fill_kernel on each
# device). Dispatch logging appears on the app's stderr in local mode.
run-2gpu-log: $(BUILD_DIR)/multi_gpu
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG_2GPU) -- $< || true

clean:
	rm -rf $(BUILD_DIR)

help:
	@echo "Multi-GPU (manual all-reduce) example - targets:"
	@echo "  make            - build multi_gpu"
	@echo "  make run-2gpu   - run on the 2-GPU config (daemon mode)"
	@echo "  make run-2gpu-log - local mode + RJ_LOG=1 (per-GPU kernel dispatches)"
	@echo "  make clean      - remove build/"
	@echo ""
	@echo "Note: RCCL collectives do NOT run under rocjitsu (inter-GPU transport"
	@echo "unsupported). This example implements all-reduce manually with basic HIP."
