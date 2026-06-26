# Example 9: PyTorch Model Debugging

## Objective

Learn to debug PyTorch deep learning models using rocjitsu:
- Run PyTorch workloads in simulation
- Debug training loops and gradients
- Identify numerical issues
- Profile GPU kernel execution
- Use daemon mode for better isolation

## What This Example Demonstrates

1. **PyTorch + rocjitsu** - Running deep learning frameworks
2. **Daemon mode** - Proper isolation for complex applications
3. **Gradient debugging** - Checking backpropagation
4. **Numerical stability** - Finding NaN/Inf issues
5. **Kernel profiling** - Identifying performance bottlenecks

## Key Debugging Points

### 1. Setting Up PyTorch with rocjitsu

**Important**: Always use daemon mode for PyTorch:
```bash
rocjitsu --daemon --config configs/amdgpu_cdna4_kmd.json -- python3 train.py
```

Daemon mode provides better isolation between PyTorch's runtime and the simulator.

### 2. Common Issues to Debug

- **NaN gradients** - Exploding/vanishing gradients
- **Incorrect loss** - Model not learning
- **Memory errors** - Out-of-memory or invalid access
- **Slow training** - Performance bottlenecks
- **Kernel failures** - Specific operation crashes

### 3. Debugging Workflow

1. Run model through rocjitsu
2. Enable logging to capture kernel launches
3. Check tensor values and gradients
4. Identify problematic operations
5. Fix and re-test

## Files

- `src/simple_model.py` - Simple neural network for debugging
- `src/gradient_check.py` - Gradient checking utilities
- `src/debug_training.py` - Training with debug hooks
- `Makefile` - Run commands for different scenarios
- `requirements.txt` - Python dependencies

## Prerequisites

```bash
# Install PyTorch with ROCm support
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Or use the rocm/pytorch Docker image (recommended)
docker run -it rocm/pytorch:latest bash
```

## Building

No compilation needed (Python scripts), but ensure dependencies are installed:

```bash
cd usage-examples/09-pytorch-model-debug
pip install -r requirements.txt
```

## Running

### Basic Model Training

```bash
make run-basic
```

Equivalent to:
```bash
rocjitsu --daemon --config ../../configs/amdgpu_cdna4_kmd.json -- \
    python3 src/simple_model.py
```

### Training with Debug Logging

```bash
make run-debug
```

Equivalent to:
```bash
RJ_LOG=1 rocjitsu --daemon --config ../../configs/amdgpu_cdna4_kmd.json -- \
    python3 src/debug_training.py
```

### Gradient Checking

```bash
make run-gradient-check
```

Runs gradient verification to ensure backpropagation is correct.

### Profile Kernel Execution

```bash
make run-profile
```

Equivalent to:
```bash
RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 RJ_SINKS=file RJ_SINK_DIR=logs \
    rocjitsu --daemon --config ../../configs/amdgpu_cdna4_kmd.json -- \
    python3 src/simple_model.py
```

## Example Scripts

### 1. simple_model.py - Basic Training

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create model and move to GPU
device = torch.device('cuda')
model = SimpleNet().to(device)

# Create dummy data
batch_size = 32
x = torch.randn(batch_size, 1, 28, 28, device=device)
y = torch.randint(0, 10, (batch_size,), device=device)

# Training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 2. debug_training.py - Debug Hooks

```python
import torch
import torch.nn as nn

def check_gradients(model, name):
    """Check for NaN or infinite gradients"""
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"[{name}] NaN gradient in {param_name}")
            if torch.isinf(param.grad).any():
                print(f"[{name}] Inf gradient in {param_name}")
            grad_norm = param.grad.norm().item()
            print(f"[{name}] {param_name} grad norm: {grad_norm:.6f}")

def register_debug_hooks(model):
    """Register forward/backward hooks for debugging"""
    
    def forward_hook(module, input, output):
        print(f"Forward: {module.__class__.__name__}")
        print(f"  Input shape: {input[0].shape}")
        print(f"  Output shape: {output.shape}")
        if torch.isnan(output).any():
            print(f"  WARNING: NaN in output!")
        if torch.isinf(output).any():
            print(f"  WARNING: Inf in output!")
    
    def backward_hook(module, grad_input, grad_output):
        print(f"Backward: {module.__class__.__name__}")
        if grad_output[0] is not None:
            print(f"  Grad output shape: {grad_output[0].shape}")
            if torch.isnan(grad_output[0]).any():
                print(f"  WARNING: NaN in gradient!")
    
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

# Usage
model = SimpleNet().to('cuda')
register_debug_hooks(model)
```

## Expected Output

### Successful Training

```
PyTorch Simple Model Training
Device: cuda
Model: SimpleNet (784 -> 128 -> 64 -> 10)

[rocjitsu] Daemon mode initialized
[rocjitsu] Config: amdgpu_cdna4_kmd.json
[rocjitsu] GPU: gfx950

Epoch 1/5
  Loss: 2.3154
  Kernels executed: 47
  
Epoch 2/5
  Loss: 2.1023
  Kernels executed: 47
  
Epoch 3/5
  Loss: 1.8567
  Kernels executed: 47
  
Epoch 4/5
  Loss: 1.6234
  Kernels executed: 47
  
Epoch 5/5
  Loss: 1.4321
  Kernels executed: 47

Training completed successfully!
Final accuracy: 45.2%
```

### With Debug Logging (RJ_LOG=1)

```
[rocjitsu] Kernel dispatch: aten::linear
[rocjitsu]   Grid: (4, 1, 1), Block: (256, 1, 1)
Forward: Linear
  Input shape: torch.Size([32, 784])
  Output shape: torch.Size([32, 128])
  
[rocjitsu] Kernel dispatch: aten::relu
[rocjitsu]   Grid: (2, 1, 1), Block: (256, 1, 1)
Forward: ReLU
  Input shape: torch.Size([32, 128])
  Output shape: torch.Size([32, 128])
  
[rocjitsu] Kernel dispatch: aten::linear
[rocjitsu]   Grid: (2, 1, 1), Block: (256, 1, 1)
Forward: Linear
  Input shape: torch.Size([32, 128])
  Output shape: torch.Size([32, 64])

... (more kernel launches)

Backward: Linear
  Grad output shape: torch.Size([32, 10])
  fc3.weight grad norm: 0.234567
  fc3.bias grad norm: 0.012345
```

## Common Issues and Solutions

### Issue 1: NaN Loss

**Symptom**: Loss becomes NaN after a few iterations

**Debug Steps**:
1. Enable debug hooks: `python3 src/debug_training.py`
2. Check for NaN in intermediate outputs
3. Look for numerical instability (division by zero, log(0), etc.)

**Common Causes**:
- Learning rate too high
- Unstable activation functions
- Missing batch normalization
- Gradient explosion

**Solution**:
```python
# Add gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Use more stable loss functions
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Lower learning rate
optimizer = optim.SGD(model.parameters(), lr=0.001)  # Was 0.01
```

### Issue 2: Gradient Check Failures

**Symptom**: Numerical gradient differs from analytical gradient

**Debug Steps**:
1. Run gradient checker: `make run-gradient-check`
2. Identify which layer fails
3. Check custom operations

**Example**:
```python
from torch.autograd import gradcheck

# Test a specific operation
input = torch.randn(20, 20, dtype=torch.double, requires_grad=True, device='cuda')
test = gradcheck(lambda x: my_custom_op(x), input, eps=1e-6, atol=1e-4)
print(f"Gradient check: {'PASSED' if test else 'FAILED'}")
```

### Issue 3: CUDA/HIP Kernel Errors

**Symptom**: "invalid configuration argument" or kernel launch failure

**Debug Steps**:
1. Enable verbose logging: `RJ_LOG=1 make run-debug`
2. Look for kernel dispatch errors
3. Check tensor shapes and dimensions

**Common Causes**:
- Invalid tensor shapes
- Wrong number of dimensions
- Device mismatch (CPU vs GPU)

### Issue 4: Slow Training

**Symptom**: Training is unexpectedly slow

**Debug Steps**:
1. Profile execution: `make run-profile`
2. Check kernel execution times
3. Look for CPU-GPU synchronization

**Optimization Tips**:
```python
# Use pin_memory for faster H2D transfers
dataloader = DataLoader(dataset, batch_size=32, pin_memory=True)

# Avoid unnecessary .item() calls (forces synchronization)
# Bad:
if loss.item() < threshold:  # Synchronizes every iteration
    
# Good:
if epoch % 10 == 0:  # Only check periodically
    if loss.item() < threshold:

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Debugging Checklist

### Before Training
- [ ] Model moves to correct device (`model.to('cuda')`)
- [ ] Input tensors on correct device
- [ ] Batch size reasonable (not too large for memory)
- [ ] Learning rate appropriate

### During Training
- [ ] Loss decreasing (or converging for GANs)
- [ ] No NaN/Inf in loss or gradients
- [ ] Gradients have reasonable magnitude (not vanishing/exploding)
- [ ] Accuracy improving (for classification)

### After Training
- [ ] Model can make predictions on new data
- [ ] Saved weights load correctly
- [ ] Performance meets expectations

## Advanced Debugging

### Memory Profiling

```python
# Track memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Profile memory during training
torch.cuda.memory._record_memory_history()
# ... run training ...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

### Kernel Timing

```bash
# Enable profiling
RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 \
RJ_SINKS=file RJ_SINK_DIR=./logs \
make run-basic

# Check logs for kernel execution times
cat logs/profile_*.txt
```

### Distributed Training Debug

For multi-GPU training with RCCL:
```bash
rocjitsu --daemon --config ../../configs/amdgpu_cdna4_kmd_2gpu.json -- \
    python3 src/distributed_train.py
```

## Key Takeaways

- Always use `--daemon` mode for PyTorch
- Enable logging (`RJ_LOG=1`) to see kernel launches
- Use debug hooks to catch NaN/Inf early
- Check gradients with `gradcheck`
- Profile to find performance bottlenecks
- Test with small batch sizes first

## Next Steps

- [Example 10: Multi-GPU RCCL](../10-multi-gpu-collective/) - Distributed training
- [Example 8: GEMM Debugging](../08-gemm-debugging/) - rocBLAS operations
- [rocjitsu Daemon Mode](../../docs/rocjitsu-cli.md#daemon-mode) - Full daemon documentation