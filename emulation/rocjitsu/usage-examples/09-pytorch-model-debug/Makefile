# Makefile for PyTorch debugging example
# Demonstrates debugging deep learning models with rocjitsu

# Python interpreter
PYTHON ?= python3

# Paths
SRC_DIR = src
ROCJITSU_ROOT = ../..
CONFIG_DIR = $(ROCJITSU_ROOT)/configs
ROCJITSU_BIN ?= $(ROCJITSU_ROOT)/build/tools/rocjitsu/rocjitsu

# Configuration
CONFIG ?= $(CONFIG_DIR)/amdgpu_cdna4_kmd.json
CONFIG_2GPU ?= $(CONFIG_DIR)/amdgpu_cdna4_kmd_2gpu.json

# Scripts
SIMPLE_MODEL = $(SRC_DIR)/simple_model.py
DEBUG_TRAINING = $(SRC_DIR)/debug_training.py
GRADIENT_CHECK = $(SRC_DIR)/gradient_check.py

.PHONY: all install run-basic run-debug run-gradient-check run-profile clean help

# Default target
all: install

# Install Python dependencies
install:
	@echo "Installing Python dependencies..."
	@echo "NOTE: Ensure PyTorch with ROCm support is installed"
	@echo "      pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
	@$(PYTHON) -c "import torch; print(f'PyTorch version: {torch.__version__}')" 2>/dev/null || \
		(echo "ERROR: PyTorch not found. Install with:"; \
		 echo "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"; \
		 exit 1)
	@$(PYTHON) -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run basic training
run-basic: $(SIMPLE_MODEL)
	@echo "Running basic PyTorch model through rocjitsu..."
	@echo "Config: $(CONFIG)"
	@echo "NOTE: Using daemon mode (recommended for PyTorch)"
	@echo ""
	$(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL)

# Run with debug logging
run-debug: $(SIMPLE_MODEL)
	@echo "Running with verbose logging..."
	@echo ""
	RJ_LOG=1 $(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL)

# Run gradient checking
run-gradient-check: $(GRADIENT_CHECK)
	@echo "Running gradient verification..."
	@echo ""
	$(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(GRADIENT_CHECK)

# Run with profiling
run-profile: $(SIMPLE_MODEL)
	@mkdir -p logs
	@echo "Running with profiling enabled..."
	@echo "Output directory: logs/"
	@echo ""
	RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 \
	RJ_SINKS=file \
	RJ_SINK_DIR=logs \
	$(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL)
	@echo ""
	@echo "Profile data saved to: logs/"
	@ls -lh logs/ 2>/dev/null || true

# Run with race detection
run-race-detect: $(SIMPLE_MODEL)
	@echo "Running with race detection..."
	@echo ""
	RJ_SINKS=race_detector $(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL)

# Run on 2-GPU configuration
run-2gpu: $(SIMPLE_MODEL)
	@echo "Running with 2-GPU configuration..."
	@echo "Config: $(CONFIG_2GPU)"
	@echo ""
	$(ROCJITSU_BIN) --daemon --config $(CONFIG_2GPU) -- $(PYTHON) $(SIMPLE_MODEL)

# Run with custom epochs
run-epochs:
	@echo "Running with custom number of epochs..."
	@echo "Usage: make run-epochs EPOCHS=10"
	$(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(SIMPLE_MODEL) $(EPOCHS)

# Run training script with debug hooks
run-debug-hooks: $(DEBUG_TRAINING)
	@echo "Running training with debug hooks..."
	@echo ""
	RJ_LOG=1 $(ROCJITSU_BIN) --daemon --config $(CONFIG) -- $(PYTHON) $(DEBUG_TRAINING)

# Test without rocjitsu (requires real GPU)
run-native: $(SIMPLE_MODEL)
	@echo "Running natively (requires real ROCm GPU)..."
	@echo ""
	$(PYTHON) $(SIMPLE_MODEL)

# Clean generated files
clean:
	@echo "Cleaning logs and cache..."
	rm -rf logs __pycache__ $(SRC_DIR)/__pycache__
	find . -name "*.pyc" -delete
	@echo "Clean complete."

# Help
help:
	@echo "PyTorch Model Debugging - Makefile Targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install           - Check Python dependencies"
	@echo ""
	@echo "Basic Running:"
	@echo "  make run-basic         - Run simple model (daemon mode)"
	@echo "  make run-debug         - Run with verbose logging"
	@echo "  make run-native        - Run without rocjitsu (needs real GPU)"
	@echo ""
	@echo "Debugging:"
	@echo "  make run-gradient-check - Verify gradients numerically"
	@echo "  make run-debug-hooks    - Run with forward/backward hooks"
	@echo "  make run-race-detect    - Enable race detection"
	@echo ""
	@echo "Profiling:"
	@echo "  make run-profile       - Profile kernel execution"
	@echo ""
	@echo "Advanced:"
	@echo "  make run-2gpu          - Use 2-GPU configuration"
	@echo "  make run-epochs EPOCHS=10 - Run custom number of epochs"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean             - Remove logs and cache files"
	@echo "  make help              - Show this help"
	@echo ""
	@echo "Environment Variables:"
	@echo "  RJ_LOG=1                    - Verbose logging"
	@echo "  RJ_SINKS=race_detector      - Enable race detection"
	@echo "  RJ_SINKS=file               - Write logs to file"
	@echo "  RJ_SINK_DIR=<dir>           - Log output directory"
	@echo ""
	@echo "Example:"
	@echo "  make run-profile        - Profile model execution"
	@echo "  make run-debug          - See all kernel launches"