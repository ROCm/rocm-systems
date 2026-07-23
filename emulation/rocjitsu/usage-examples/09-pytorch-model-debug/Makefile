# Makefile for the PyTorch inference example under rocjitsu.
# Training (backward + optimizer) is impractically slow under the functional
# emulator, so this example runs a small forward pass (inference) only.

PYTHON ?= python3
SRC_DIR = src
ROCJITSU_ROOT = ../..
CONFIG_DIR = $(ROCJITSU_ROOT)/configs
ROCJITSU_BIN ?= $(ROCJITSU_ROOT)/build/tools/rocjitsu/rocjitsu
CONFIG ?= $(CONFIG_DIR)/amdgpu_cdna4_kmd.json

SIMPLE_MODEL = $(SRC_DIR)/simple_model.py

.PHONY: all check run run-log run-native clean help

all: check

# Check that a ROCm-enabled PyTorch is present (needs torch+rocm, not torch+cpu).
check:
	@$(PYTHON) -c "import torch; print('PyTorch:', torch.__version__); print('cuda_available:', torch.cuda.is_available())" 2>/dev/null || \
		(echo "ERROR: PyTorch not found. Install a ROCm wheel, e.g.:"; \
		 echo "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.3"; \
		 exit 1)

# Recommended: daemon mode. Completes reliably; the model runs on the emulated
# GPU and host-side hooks check for NaN/Inf. (Emulation makes this take ~1 min.)
run: $(SIMPLE_MODEL)
	$(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL)

# Inspect the dispatched GPU kernels (and 'mfma detected' for the matmul).
# RJ_LOG dispatch output is emitted on stderr in LOCAL mode (daemon mode routes
# it to the daemon). NOTE: local mode is less isolated and a full framework may
# abort partway — you still see the dispatch log and MFMA confirmation.
run-log: $(SIMPLE_MODEL)
	RJ_LOG=1 $(ROCJITSU_BIN) --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL) || true

# Run natively on a real ROCm GPU (no emulation) — useful for full training,
# which is not practical under rocjitsu.
run-native: $(SIMPLE_MODEL)
	$(PYTHON) $(SIMPLE_MODEL)

clean:
	rm -rf logs __pycache__ $(SRC_DIR)/__pycache__
	find . -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "PyTorch inference under rocjitsu - targets:"
	@echo "  make check       - verify a ROCm PyTorch (torch+rocm) is installed"
	@echo "  make run         - run inference in daemon mode (completes; host NaN check)"
	@echo "  make run-log     - local mode + RJ_LOG=1: show kernel dispatches + 'mfma detected'"
	@echo "  make run-native  - run on a real GPU without rocjitsu"
	@echo "  make clean       - remove caches"
	@echo ""
	@echo "Notes:"
	@echo "  * Needs a ROCm PyTorch wheel (torch+rocm); a torch+cpu wheel does not"
	@echo "    exercise the emulator (cuda_available would be False)."
	@echo "  * Training (backward/optimizer) is impractically slow under the"
	@echo "    functional emulator; this example does inference only."
	@echo "  * NaN/Inf detection is host-side PyTorch (torch.isnan), not a rocjitsu"
	@echo "    feature. RJ_LOG shows kernel dispatches + MFMA usage."
