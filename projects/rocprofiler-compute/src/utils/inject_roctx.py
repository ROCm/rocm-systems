# ruff: noqa
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################


"""
ROCTX Injection Wrapper - Auto-discovers and intercepts ALL PyTorch operators
Usage: python inject_roctx.py main.py --epochs 1 --batch-size 4
"""

import os
import sys
from pathlib import Path

# Add parent directory to Python path for config module
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
# Attempt to load ROCTX module from ROCm installation
rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
candidate_paths = [
    f"{rocm_root}/lib/{python_version}/site-packages",
    f"{rocm_root}/libexec/rocprofiler-sdk/python",
]

for candidate in candidate_paths:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from utils.logger import console_error, console_log, console_warning

console_log("torch trace", f"Python version: {python_version}")

try:
    from roctx import rangePop, rangePush

    roctx_path = Path(rangePush.__code__.co_filename).parent

    console_log(
        "torch trace",
        f"ROCTX module loaded from: {roctx_path}",
    )
except ImportError:
    console_error(
        f"Looked for Roctx in :{candidate_paths}"
        "ROCTX Python module not found.\n"
        "Please ensure that the rocprofiler-sdk is installed"
        " and that the roctx Python bindings are available for your Python version."
        "You may need to reinstall or rebuild ROCm for your current Python environment.\n"
        "The --torch-trace option requires a valid roctx installation.\n",
    )
    sys.exit(1)

try:
    import torch
    import torch._C

    console_log(f"PyTorch version: {torch.__version__}")
except ImportError:
    console_warning(
        "PyTorch is not installed or not properly configured.\n"
        "The --torch-trace option requires a valid PyTorch installation.\n"
        "Please install PyTorch and try again."
    )
    sys.exit(0)

import importlib.util
import inspect
from functools import wraps

import torch.nn.functional as F

import threading

# Thread-local stacks for hierarchical marker names
_thread_local = threading.local()


def get_marker_stack():
    if not hasattr(_thread_local, "marker_stack"):
        _thread_local.marker_stack = []
    return _thread_local.marker_stack


def get_context_stack():
    if not hasattr(_thread_local, "context_stack"):
        _thread_local.context_stack = []
    return _thread_local.context_stack


if hasattr(torch._C, "_dispatch_call"):
    original_dispatch_call = torch._C._dispatch_call

    def dispatch_call_with_roctx(*args, **kwargs):
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        op_name = str(args[0]) if args else "aten_op"
        marker_stack.append(f"aten::{op_name}")
        # Use call count and caller location for context, similar to other wrappers
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"
        # Optionally, use a call counter (not strictly necessary, but for consistency)
        if not hasattr(dispatch_call_with_roctx, "_call_count"):
            dispatch_call_with_roctx._call_count = 0
        dispatch_call_with_roctx._call_count += 1
        context_stack.append(f"#{dispatch_call_with_roctx._call_count}@{location}")
        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)
        rangePush(full_marker_name)
        # using try / finally to ensure rangePop is called.
        try:
            return original_dispatch_call(*args, **kwargs)
        except Exception as e:
            console_warning("torch trace", f"Error in {full_marker_name}: {e}")
            console_warning(
                "torch trace", "Cannot inject ROCTX markers for torch._C._dispatch_call"
            )
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    torch._C._dispatch_call = dispatch_call_with_roctx
else:
    console_log("torch._C._dispatch_call not found; skipping dispatcher patching.")

try:
    import torchvision

    original_tv_dispatch_call = torchvision._C._dispatch_call

    def tv_dispatch_call_with_roctx(*args, **kwargs):
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        op_name = str(args[0]) if args else "vision_op"
        marker_stack.append(f"torchvision::{op_name}")

        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        if not hasattr(tv_dispatch_call_with_roctx, "_call_count"):
            tv_dispatch_call_with_roctx._call_count = 0
        tv_dispatch_call_with_roctx._call_count += 1
        context_stack.append(f"#{tv_dispatch_call_with_roctx._call_count}@{location}")

        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)
        rangePush(full_marker_name)
        try:
            return original_tv_dispatch_call(*args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    torchvision._C._dispatch_call = tv_dispatch_call_with_roctx
except Exception:
    pass  # torchvision not installed or no _C._dispatch_call

try:
    import torch.distributed as dist

    original_all_reduce = dist.all_reduce

    def all_reduce_with_roctx(*args, **kwargs):
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        marker_stack.append("torch.distributed.all_reduce")

        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        if not hasattr(all_reduce_with_roctx, "_call_count"):
            all_reduce_with_roctx._call_count = 0
        all_reduce_with_roctx._call_count += 1
        context_stack.append(f"#{all_reduce_with_roctx._call_count}@{location}")

        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)
        rangePush(full_marker_name)
        try:
            return original_all_reduce(*args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    dist.all_reduce = all_reduce_with_roctx
except Exception as e:
    console_warning("torch trace", f"Could not patch torch.distributed.all_reduce: {e}")

try:
    import torch.cuda

    original_set_device = torch.cuda.set_device

    def set_device_with_roctx(*args, **kwargs):
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        marker_stack.append("torch.cuda.set_device")

        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        if not hasattr(set_device_with_roctx, "_call_count"):
            set_device_with_roctx._call_count = 0
        set_device_with_roctx._call_count += 1
        context_stack.append(f"#{set_device_with_roctx._call_count}@{location}")

        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)
        rangePush(full_marker_name)
        try:
            return original_set_device(*args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    torch.cuda.set_device = set_device_with_roctx
except Exception:
    pass


def roctx_wrapper(func, name=None):
    func_name = name or func.__name__
    call_counter = {"count": 0}

    @wraps(func)
    def wrapper(*args, **kwargs):
        call_counter["count"] += 1
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        # Build hierarchical marker name with separate context
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        marker_stack.append(func_name)
        context_stack.append(f"#{call_counter['count']}@{location}")
        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)

        rangePush(full_marker_name)
        try:
            result = func(*args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()
        return result

    return wrapper


def auto_discover_torch_callables(module, prefix, exclude_patterns=None):
    """Automatically discover all callable functions in a module."""
    if exclude_patterns is None:
        exclude_patterns = ["__", "_", "is_", "set_", "get_"]

    functions = {}
    for name in dir(module):
        # Skip private/internal functions
        if any(name.startswith(pat) for pat in exclude_patterns):
            continue

        try:
            attr = getattr(module, name)
            # Only wrap callables (functions, not classes or constants)
            if callable(attr) and not isinstance(attr, type):
                full_name = f"{prefix}.{name}"
                functions[full_name] = (module, name, attr)
        except Exception as e:
            console_warning(type(e))
            console_warning(f"Could not access {prefix}.{name}: {e}")

    return functions


def inject_roctx_into_torch():
    """Monkey-patch PyTorch operations to add ROCTX markers."""

    console_log("Auto-discovering PyTorch operations to wrap...")

    # Auto-discover functions from key modules
    all_operations = {}

    # torch.* functions (matmul, mm, cat, etc.)
    all_operations.update(auto_discover_torch_callables(torch, "torch"))

    # torch.nn.functional.* functions (linear, relu, softmax, etc.)
    all_operations.update(auto_discover_torch_callables(F, "torch.nn.functional"))

    # torch.linalg.* functions (matrix operations)
    try:
        all_operations.update(
            auto_discover_torch_callables(torch.linalg, "torch.linalg")
        )
    except Exception as e:
        console_warning(type(e))
        console_warning(f"Could not access torch.linalg: {e}")

    # torch.fft.* functions (FFT operations)
    try:
        all_operations.update(auto_discover_torch_callables(torch.fft, "torch.fft"))
    except Exception as e:
        console_warning(type(e))
        console_warning(f"Could not access torch.fft: {e}")
    console_log(f"Found {len(all_operations)} operations to wrap")
    console_log("Injecting ROCTX markers into PyTorch operations...")

    wrapped_count = 0
    failed_count = 0

    for full_name, (module, attr_name, original_func) in all_operations.items():
        try:
            # Replace with wrapped version
            wrapped_func = roctx_wrapper(original_func, full_name)
            setattr(module, attr_name, wrapped_func)
            wrapped_count += 1

            # Print first 20 and last 5 for visibility
            if wrapped_count <= 20 or wrapped_count > len(all_operations) - 5:
                console_log(f"Wrapped: {full_name}")
            elif wrapped_count == 21:
                console_log(
                    f"  ... (wrapping {len(all_operations) - 25} more operations)"
                )

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Only show first few failures
                console_warning(f"Failed to wrap {full_name}: {e}")

    # Wrap tensor methods
    original_backward = torch.Tensor.backward
    backward_counter = {"count": 0}

    def backward_with_roctx(self, *args, **kwargs):
        backward_counter["count"] += 1
        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        marker_stack.append("torch.Tensor.backward")
        context_stack.append(f"#{backward_counter['count']}@{location}")
        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)

        rangePush(full_marker_name)
        try:
            return original_backward(self, *args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    torch.Tensor.backward = backward_with_roctx
    wrapped_count += 1
    console_log("Wrapped: torch.Tensor.backward")

    console_log(f"Wrapped {wrapped_count} operations with ROCTX markers")
    if failed_count > 0:
        console_warning(
            f"Failed to wrap {failed_count} operations (likely not patchable)"
        )


def inject_roctx_into_optimizer():
    """Wrap optimizer step() method."""
    from torch.optim import Optimizer

    original_step = Optimizer.step

    def step_with_roctx(self, *args, **kwargs):
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        marker_stack.append(f"optimizer.{self.__class__.__name__}.step")

        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        if not hasattr(self, "_roctx_step_call_count"):
            self._roctx_step_call_count = 0
        self._roctx_step_call_count += 1
        context_stack.append(f"#{self._roctx_step_call_count}@{location}")

        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)
        rangePush(full_marker_name)
        try:
            return original_step(self, *args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    Optimizer.step = step_with_roctx
    console_log("Wrapped optimizer.step() with ROCTX markers\n")


def inject_roctx_into_model():
    """Wrap nn.Module forward() method with call counter."""

    from torch import nn

    original_call = nn.Module.__call__

    # Per-instance call counters
    def call_with_roctx(self, *args, **kwargs):
        class_name = self.__class__.__name__
        if not hasattr(self, "_roctx_call_count"):
            self._roctx_call_count = 0
        self._roctx_call_count += 1

        current_frame = inspect.currentframe()
        caller_frame = current_frame.f_back if current_frame is not None else None
        if caller_frame is not None:
            filename = caller_frame.f_code.co_filename
            location = f"{Path(filename).name}:{caller_frame.f_lineno}"
        else:
            location = "unknown:0"

        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        marker_stack.append(f"nn.Module.{class_name}.forward")
        context_stack.append(f"#{self._roctx_call_count}@{location}")
        full_marker_name = "/".join(marker_stack) + ":" + "/".join(context_stack)

        rangePush(full_marker_name)
        try:
            return original_call(self, *args, **kwargs)
        finally:
            rangePop()
            marker_stack.pop()
            context_stack.pop()

    nn.Module.__call__ = call_with_roctx
    console_log("Wrapped nn.Module forward() with ROCTX markers\n")


def instrument_all_torch_ops():
    # Base operator namespaces to instrument across backends
    op_namespaces = ["aten", "quantized", "ml", "prims"]
    # Only include NVIDIA-specific nvfuser ops when running on a CUDA build
    torch_version = getattr(torch, "version", None)
    cuda_version = (
        getattr(torch_version, "cuda", None) if torch_version is not None else None
    )
    hip_version = (
        getattr(torch_version, "hip", None) if torch_version is not None else None
    )
    if cuda_version is not None and hip_version is None:
        op_namespaces.append("nvfuser")
    wrapped_count = 0
    failed_count = 0

    console_log("Instrumenting torch.ops namespace for C++ ATen operations...")
    for ns in op_namespaces:
        ops = getattr(torch.ops, ns, None)
        if ops is None:
            continue
        ns_count = 0
        for op_name in dir(ops):
            if op_name.startswith("__"):
                continue
            try:
                op = getattr(ops, op_name, None)
                if not callable(op):
                    continue
                if hasattr(op, "_roctx_wrapped"):
                    continue

                def make_wrapper(original_op, namespace, operation_name):
                    def wrapper(*args, **kwargs):
                        marker_name = f"torch.ops.{namespace}.{operation_name}"
                        marker_stack = get_marker_stack()
                        context_stack = get_context_stack()
                        marker_stack.append(marker_name)
                        filtered_context = [c for c in context_stack if c]
                        full_marker_name = "/".join(marker_stack) + (
                            ":" + "/".join(filtered_context) if filtered_context else ""
                        )

                        rangePush(full_marker_name)
                        try:
                            return original_op(*args, **kwargs)
                        finally:
                            rangePop()
                            marker_stack.pop()

                    wrapper._roctx_wrapped = True
                    return wrapper

                wrapped_op = make_wrapper(op, ns, op_name)
                setattr(ops, op_name, wrapped_op)
                wrapped_count += 1
                ns_count += 1
            except Exception as e:
                failed_count += 1
                if failed_count <= 10:
                    console_log(f"  Failed to wrap {ns}.{op_name}: {e}")

        if ns_count > 0:
            console_log(f"  Wrapped {ns_count} operations in torch.ops.{ns}")

    console_log(f"[ROCTX] Total: {wrapped_count} torch.ops operations wrapped")
    if failed_count > 0:
        console_log(f"[ROCTX] ({failed_count} operations failed to wrap)")
    console_log("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console_log("Usage: python inject_roctx.py <script.py> [script_args...]")
        sys.exit(1)

    # Get target script and its arguments
    target_script = sys.argv[1]
    script_args = sys.argv[2:]

    # Inject ROCTX markers BEFORE importing the target script
    inject_roctx_into_torch()
    inject_roctx_into_optimizer()
    inject_roctx_into_model()
    instrument_all_torch_ops()

    console_log("=" * 70)
    console_log("Starting target script with ROCTX instrumentation...")
    console_log("=" * 70)

    # Modify sys.argv so the target script sees correct arguments
    sys.argv = [target_script] + script_args

    # Load and execute the target script
    spec = importlib.util.spec_from_file_location("__main__", target_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = module
    spec.loader.exec_module(module)
