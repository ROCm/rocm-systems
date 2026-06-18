# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Instrument PyTorch with ROCTX ranges for rocprof-compute --torch-trace.

Loaded by rocprofv3 as ``python inject_roctx.py main.py [args...]``.
ATen operator coverage uses the C++ RecordFunction extension when
available, with a Python ``TorchDispatchMode`` fallback. Structural
wrappers for ``nn.Module``, ``Optimizer``, distributed collectives,
CUDA graphs, Triton, and ``torch.compile`` run regardless of tier.
"""

import importlib
import importlib.util
import inspect
import os
import sys
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

# Ensure the sibling 'utils' package is importable when rocprofv3 launches
# this script directly.
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))

rocm_root = os.environ.get("ROCM_PATH", "/opt/rocm")
python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
candidate_paths = [
    f"{rocm_root}/lib/{python_version}/site-packages",
    f"{rocm_root}/libexec/rocprofiler-sdk/python",
]

for candidate in candidate_paths:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from utils.logger import console_error, console_log, console_warning  # noqa: E402

console_log("torch trace", f"Workload Python Version: {python_version}")

try:
    from roctx import rangePop, rangePush

    if hasattr(rangePush, "__code__") and hasattr(rangePush.__code__, "co_filename"):
        roctx_path = Path(rangePush.__code__.co_filename).parent
    else:
        roctx_path = "<unknown>"

    console_log("torch trace", f"ROCTX module loaded from: {roctx_path}")
except ImportError:
    console_error(
        "torch trace",
        f"Looked for roctx in: {candidate_paths}\n"
        "ROCTX not found. --torch-trace requires roctx from rocprofiler-sdk. "
        "Ensure your workload uses a Python version for which "
        "roctx bindings are available in your ROCm installation.\n",
    )
    sys.exit(1)

_TORCH_ROOT = ""

try:
    import torch
    import torch._C  # noqa: F401

    console_log("torch trace", f"PyTorch version: {torch.__version__}")
    try:
        _TORCH_ROOT = str(Path(torch.__file__).resolve().parent) + os.sep
    except Exception:
        _TORCH_ROOT = ""
except ImportError:
    console_warning(
        "torch trace",
        "PyTorch is not installed or not properly configured.\n"
        "The --torch-trace option requires a valid PyTorch installation.\n"
        "Please install PyTorch and try again.",
    )
    sys.exit(0)

# Optional dependencies. Each is bound to None on import failure so the
# corresponding patch site can early-return.
try:
    import torch.distributed as dist
except Exception:
    dist = None

try:
    import torch.distributed._functional_collectives as fc
except Exception:
    fc = None

try:
    from torch.distributed.distributed_c10d import ProcessGroup
except Exception:
    ProcessGroup = None

try:
    import torch.cuda as cuda_mod
except Exception:
    cuda_mod = None

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except Exception:
    TorchDispatchMode = None

try:
    from torch.optim import Optimizer
except Exception:
    Optimizer = None

try:
    from torch.autograd import Function
except Exception:
    Function = None

try:
    from torch import nn
except Exception:
    nn = None

try:
    from triton.compiler import CompiledKernel
except Exception:
    CompiledKernel = None

try:
    from utils import inject_roctx_loader as _roctx_loader_module
    from utils.inject_roctx_loader import load as _load_roctx_recordfn
except Exception as _e:
    console_warning(
        "torch trace",
        f"inject_roctx_loader unavailable; falling back to Python tier: {_e}",
    )
    _load_roctx_recordfn = None
    _roctx_loader_module = None

_roctx_recordfn = None
_USING_C_TIER = False
_C_TIER_INITIALIZED = False


def _initialize_c_tier() -> bool:
    """Load and install the roctx_recordfn extension once per process."""
    global _roctx_recordfn, _USING_C_TIER, _C_TIER_INITIALIZED

    if _C_TIER_INITIALIZED:
        return _USING_C_TIER

    _C_TIER_INITIALIZED = True

    if _load_roctx_recordfn is None:
        _roctx_recordfn = None
        _USING_C_TIER = False
        return False

    try:
        _roctx_recordfn = _load_roctx_recordfn()
    except Exception as _e:
        console_warning(
            "torch trace",
            f"loader raised; falling back to Python tier: {_e}",
        )
        _roctx_recordfn = None

    if _roctx_recordfn is not None:
        try:
            _roctx_recordfn.install()
            console_log(
                "torch trace",
                "Coverage tier: C++ RecordFunction (global callback).",
            )
            _USING_C_TIER = True
            return True
        except Exception as _e:
            console_warning(
                "torch trace",
                f".so install() raised; falling back to Python tier: {_e}",
            )
            _roctx_recordfn = None

    _USING_C_TIER = False
    return False


def _emit_python_tier_fallback_warning() -> None:
    """Emit one warning when the C++ tier is unavailable."""
    if _USING_C_TIER:
        return
    loader_trail = ""
    if _roctx_loader_module is not None:
        try:
            _tier, diagnostics = _roctx_loader_module.consume_diagnostics()
            loader_trail = _roctx_loader_module.format_load_diagnostic_trail(
                diagnostics,
            )
        except Exception:
            loader_trail = ""
    trail_block = f"\nLoader trail:\n{loader_trail}" if loader_trail else ""
    console_warning(
        "torch trace",
        "Coverage tier: Python-only injector. Autograd backward markers "
        "will carry single-level labels and the main-thread USER_SCOPE "
        "chain will not be visible from backward Nodes." + trail_block,
    )


# Per-thread marker, context, and tier stacks.
_thread_local = threading.local()

# Retained at module scope so it outlives install_dispatcher_hook().
_active_dispatch_mode = None


def get_marker_stack() -> list[str]:
    if not hasattr(_thread_local, "marker_stack"):
        _thread_local.marker_stack = []
    return _thread_local.marker_stack


def get_context_stack() -> list[str]:
    if not hasattr(_thread_local, "context_stack"):
        _thread_local.context_stack = []
    return _thread_local.context_stack


def get_tier_stack() -> list[bool]:
    # True entries denote C++ pushes; False entries denote Python pushes.
    if not hasattr(_thread_local, "tier_stack"):
        _thread_local.tier_stack = []
    return _thread_local.tier_stack


def resolve_user_caller_location() -> str:
    """Return ``"file:line"`` for the nearest user frame, or ``"python.dispatch:0"``."""
    this_file = __file__
    frame = inspect.currentframe()
    while frame is not None:
        fn_path = frame.f_code.co_filename
        if fn_path != this_file and (
            not _TORCH_ROOT or not fn_path.startswith(_TORCH_ROOT)
        ):
            return f"{Path(fn_path).name}:{frame.f_lineno}"
        frame = frame.f_back
    return "python.dispatch:0"


def _push_scope(marker: str, context: str) -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    tier_stack = get_tier_stack()

    used_cpp = False
    if _USING_C_TIER:
        try:
            _roctx_recordfn.push_user_scope(marker, context)
            used_cpp = True
        except Exception:
            pass

    if not used_cpp:
        full = (
            "/".join([*marker_stack, marker])
            + ":"
            + "/".join([*context_stack, context])
        )
        rangePush(full)

    tier_stack.append(used_cpp)
    marker_stack.append(marker)
    context_stack.append(context)


def _pop_scope() -> None:
    marker_stack = get_marker_stack()
    context_stack = get_context_stack()
    tier_stack = get_tier_stack()

    if not tier_stack:
        return

    used_cpp = tier_stack.pop()
    try:
        if used_cpp:
            _roctx_recordfn.pop_user_scope()
        else:
            rangePop()
    finally:
        if marker_stack:
            marker_stack.pop()
        if context_stack:
            context_stack.pop()


def roctx_wrapper(
    func: Callable[..., Any],
    name: Optional[str] = None,
) -> Callable[..., Any]:
    """Wrap ``func`` so each call emits a ROCTX range. Idempotent."""
    if getattr(func, "_roctx_wrapped", False):
        return func
    func_name = name or func.__name__
    call_counter = {"count": 0}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope(func_name, f"#{call_counter['count']}@{location}")
        try:
            return func(*args, **kwargs)
        finally:
            _pop_scope()

    wrapper._roctx_wrapped = True
    return wrapper


def _marker_only_init_wrapper(name: str) -> Callable[..., Any]:
    """Return an ``__init__`` that emits a ROCTX range and calls ``object.__init__``.

    Used for classes whose real construction happens in ``__new__``.
    """
    call_counter = {"count": 0}

    def marker_only_init(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        call_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope(name, f"#{call_counter['count']}@{location}")
        try:
            return object.__init__(self)
        finally:
            _pop_scope()

    marker_only_init._roctx_wrapped = True
    return marker_only_init


def _walk_subclasses(cls: type, fn: Callable[[type], None]) -> None:
    """Apply ``fn`` to every transitive subclass of ``cls``."""
    for sub in cls.__subclasses__():
        fn(sub)
        _walk_subclasses(sub, fn)


# Distributed collective entry points wrapped on torch.distributed.
DISTRIBUTED_COLLECTIVE_NAMES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_gather_object",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "broadcast",
    "broadcast_object_list",
    "reduce",
    "gather",
    "gather_object",
    "scatter",
    "scatter_object_list",
    "all_to_all",
    "all_to_all_single",
    "send",
    "recv",
    "isend",
    "irecv",
    "barrier",
    "monitored_barrier",
)

# ProcessGroup methods wrapped on every subclass.
PROCESS_GROUP_METHODS = (
    "allreduce",
    "allgather",
    "allgather_base",
    "_allgather_base",
    "reduce_scatter",
    "_reduce_scatter_base",
    "reduce_scatter_tensor",
    "alltoall",
    "alltoall_base",
    "broadcast",
    "reduce",
    "gather",
    "scatter",
    "send",
    "recv",
    "barrier",
)


def patch_distributed_collectives() -> None:
    """Wrap ``torch.distributed`` collectives and ``_functional_collectives``."""
    if dist is None:
        console_warning(
            "torch trace",
            "torch.distributed not importable; collectives will not be marked.",
        )
        return

    wrapped = []
    for fn_name in DISTRIBUTED_COLLECTIVE_NAMES:
        fn = getattr(dist, fn_name, None)
        if fn is None or not callable(fn):
            continue
        if getattr(fn, "_roctx_wrapped", False):
            continue
        try:
            setattr(dist, fn_name, roctx_wrapper(fn, f"torch.distributed.{fn_name}"))
            wrapped.append(fn_name)
        except Exception as e:
            console_warning(
                "torch trace",
                f"Could not patch torch.distributed.{fn_name}: {e}",
            )

    if fc is not None:
        for fn_name in dir(fc):
            if fn_name.startswith("_"):
                continue
            fn = getattr(fc, fn_name, None)
            if not callable(fn):
                continue
            if inspect.isclass(fn):
                continue
            if getattr(fn, "__module__", "") != fc.__name__:
                continue
            try:
                setattr(
                    fc,
                    fn_name,
                    roctx_wrapper(
                        fn, f"torch.distributed._functional_collectives.{fn_name}"
                    ),
                )
                wrapped.append(f"_functional_collectives.{fn_name}")
            except Exception as e:
                console_warning(
                    "torch trace",
                    f"Could not patch _functional_collectives.{fn_name}: {e}",
                )

    if wrapped:
        console_log(
            "torch trace",
            f"Wrapped {len(wrapped)} torch.distributed collectives with ROCTX markers",
        )


def patch_process_group_methods() -> None:
    """Wrap ``ProcessGroup`` methods on every subclass, including future ones."""
    if ProcessGroup is None:
        return

    wrapped_classes = set()
    wrapped_method_count = {"count": 0}

    def _wrap_one(cls: type) -> None:
        if cls in wrapped_classes:
            return
        wrapped_classes.add(cls)
        for method_name in PROCESS_GROUP_METHODS:
            fn = cls.__dict__.get(method_name)
            if fn is None or not callable(fn):
                continue
            if getattr(fn, "_roctx_wrapped", False):
                continue
            try:
                marker = f"ProcessGroup.{cls.__name__}.{method_name}"
                wrapped = roctx_wrapper(fn, marker)
                setattr(cls, method_name, wrapped)
                wrapped_method_count["count"] += 1
            except Exception as e:
                console_warning(
                    "torch trace",
                    f"Could not patch ProcessGroup.{cls.__name__}.{method_name}: {e}",
                )

    _wrap_one(ProcessGroup)
    _walk_subclasses(ProcessGroup, _wrap_one)

    existing_isc = ProcessGroup.__init_subclass__
    existing_isc_fn = getattr(existing_isc, "__func__", existing_isc)
    if not getattr(existing_isc_fn, "_roctx_pg_subclass_hook", False):
        original_init_subclass = existing_isc

        def init_subclass_hook(cls: type, **kwargs: Any) -> None:
            try:
                original_init_subclass_fn = getattr(
                    original_init_subclass,
                    "__func__",
                    original_init_subclass,
                )
                original_init_subclass_fn(cls, **kwargs)
            except Exception:
                pass
            try:
                _wrap_one(cls)
            except Exception as e:
                console_warning(
                    "torch trace",
                    f"_wrap_one({cls.__name__}) failed in "
                    f"ProcessGroup.__init_subclass__: {e}",
                )

        init_subclass_hook._roctx_pg_subclass_hook = True
        try:
            ProcessGroup.__init_subclass__ = classmethod(init_subclass_hook)
        except Exception:
            pass

    if wrapped_method_count["count"]:
        console_log(
            "torch trace",
            f"Wrapped {wrapped_method_count['count']} ProcessGroup methods across "
            f"{len(wrapped_classes)} subclasses with ROCTX markers",
        )


def patch_cuda_graph() -> None:
    """Wrap ``CUDAGraph.capture_begin``, ``capture_end``, and ``replay``."""
    if cuda_mod is None:
        return

    cls = getattr(cuda_mod, "CUDAGraph", None)
    if cls is None:
        return

    wrapped_methods = []
    for method_name in ("capture_begin", "capture_end", "replay"):
        fn = cls.__dict__.get(method_name)
        if fn is None or not callable(fn):
            continue
        if getattr(fn, "_roctx_wrapped", False):
            continue
        try:
            marker = f"torch.cuda.CUDAGraph.{method_name}"
            wrapped = roctx_wrapper(fn, marker)
            setattr(cls, method_name, wrapped)
            wrapped_methods.append(method_name)
        except Exception as e:
            console_warning(
                "torch trace",
                f"Could not patch CUDAGraph.{method_name}: {e}",
            )

    if wrapped_methods:
        console_log(
            "torch trace",
            "Wrapped CUDAGraph methods with ROCTX markers: "
            f"{', '.join(wrapped_methods)}",
        )


def patch_triton_launcher() -> None:
    """Wrap ``triton.compiler.CompiledKernel.__call__``."""
    if CompiledKernel is None:
        return

    original_call = getattr(CompiledKernel, "__call__", None)
    if original_call is None:
        return
    if getattr(original_call, "_roctx_wrapped", False):
        return

    @wraps(original_call)
    def call_with_roctx(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        kernel_name = (
            getattr(self, "name", None)
            or getattr(self, "metadata", None)
            or "<triton_kernel>"
        )
        if isinstance(kernel_name, dict):
            kernel_name = kernel_name.get("name", "<triton_kernel>")
        location = resolve_user_caller_location()
        marker = f"triton.CompiledKernel.{kernel_name}"
        _push_scope(marker, f"#1@{location}")
        try:
            return original_call(self, *args, **kwargs)
        finally:
            _pop_scope()

    call_with_roctx._roctx_wrapped = True
    try:
        CompiledKernel.__call__ = call_with_roctx
        console_log(
            "torch trace", "Wrapped triton.CompiledKernel.__call__ with ROCTX markers"
        )
    except Exception as e:
        console_warning(
            "torch trace",
            f"Could not patch triton.CompiledKernel.__call__: {e}",
        )


def patch_compile_callable() -> None:
    """Wrap ``torch.compile`` and the callable it returns."""
    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return
    if getattr(original_compile, "_roctx_wrapped", False):
        return

    @wraps(original_compile)
    def compile_with_roctx(
        model_or_fn: Any = None,  # noqa: ANN401
        *args: Any,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        location = resolve_user_caller_location()
        _push_scope("torch.compile", f"#1@{location}")
        try:
            compiled = original_compile(model_or_fn, *args, **kwargs)
        finally:
            _pop_scope()

        if compiled is None or not callable(compiled):
            return compiled

        fn_label = getattr(model_or_fn, "__name__", None) or type(model_or_fn).__name__

        @wraps(compiled)
        def invocation_wrapper(*c_args: Any, **c_kwargs: Any) -> Any:  # noqa: ANN401
            loc = resolve_user_caller_location()
            _push_scope(f"torch.compile.{fn_label}", f"#1@{loc}")
            try:
                return compiled(*c_args, **c_kwargs)
            finally:
                _pop_scope()

        invocation_wrapper._roctx_wrapped = True
        return invocation_wrapper

    compile_with_roctx._roctx_wrapped = True
    try:
        torch.compile = compile_with_roctx
        console_log(
            "torch trace",
            "Wrapped torch.compile + its returned callable with ROCTX markers",
        )
    except Exception as e:
        console_warning(
            "torch trace",
            f"Could not patch torch.compile invocation wrapper: {e}",
        )


def next_dispatcher_index(op_name: str) -> int:
    """Return the next per-thread occurrence index for ``op_name``."""
    counters = getattr(_thread_local, "dispatcher_counters", None)
    if counters is None:
        counters = {}
        _thread_local.dispatcher_counters = counters
    counters[op_name] = counters.get(op_name, 0) + 1
    return counters[op_name]


def warn_dispatcher_failure_once(phase: str, error: Exception) -> None:
    """Emit one warning per ``(thread, phase)`` and suppress further occurrences."""
    flag_attr = f"warned_dispatcher_failure_{phase}"
    if getattr(_thread_local, flag_attr, False):
        return
    setattr(_thread_local, flag_attr, True)
    try:
        console_warning(
            "torch trace",
            f"Dispatcher {phase} raised ({type(error).__name__}: {error}). "
            "Subsequent failures on this thread will be suppressed.",
        )
    except Exception:
        pass


def dispatcher_marker_name_for(func: Callable[..., Any]) -> str:
    """Return the marker name for a dispatcher ``func``."""
    try:
        packet = getattr(func, "overloadpacket", None) or getattr(
            func, "_overloadpacket", None
        )
        if packet is not None:
            qualified = getattr(packet, "_qualified_op_name", None)
            raw = qualified if qualified else str(packet)
        else:
            raw = str(func)
            if "::" in raw:
                ns_part, _, op_overload = raw.partition("::")
                op_part = op_overload.split(".", 1)[0]
                raw = f"{ns_part}::{op_part}"
    except Exception:
        return "<unknown_op>"

    if "::" not in raw and "." in raw:
        raw = raw.replace(".", "::", 1)

    if raw.startswith("aten::"):
        return f"torch.ops.aten.{raw[len('aten::') :]}"
    return raw


def install_dispatcher_hook() -> str:
    """Install per-op coverage. No-op on the C++ tier."""
    if _USING_C_TIER:
        console_log(
            "torch trace",
            "Operator coverage: C++ RecordFunction callback.",
        )
        return "c_tier"

    if TorchDispatchMode is None:
        console_warning(
            "torch trace",
            "TorchDispatchMode is not importable on this PyTorch build; "
            "per-op coverage will be missing.",
        )
        return "none"

    global _active_dispatch_mode

    def start_disp(op_name: str) -> None:
        idx = next_dispatcher_index(op_name)
        location = resolve_user_caller_location()
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        full_marker = (
            "/".join([*marker_stack, op_name])
            + ":"
            + "/".join([*context_stack, f"#{idx}@{location}"])
        )
        rangePush(full_marker)
        marker_stack.append(op_name)
        context_stack.append(f"#{idx}@{location}")

    def end_disp() -> None:
        marker_stack = get_marker_stack()
        context_stack = get_context_stack()
        try:
            rangePop()
        finally:
            if marker_stack:
                marker_stack.pop()
            if context_stack:
                context_stack.pop()

    class RoctxDispatchMode(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func: Callable[..., Any],
            types: Any,  # noqa: ANN401
            args: Any = (),  # noqa: ANN401
            kwargs: Any = None,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            kwargs = kwargs or {}
            op_name = dispatcher_marker_name_for(func)
            pushed = False
            try:
                start_disp(op_name)
                pushed = True
            except Exception as e:
                warn_dispatcher_failure_once("start", e)
            try:
                return func(*args, **kwargs)
            finally:
                if pushed:
                    try:
                        end_disp()
                    except Exception as e:
                        warn_dispatcher_failure_once("end", e)

    try:
        mode = RoctxDispatchMode()
        mode.__enter__()
    except Exception as e:
        console_warning("torch trace", f"TorchDispatchMode activation failed: {e}")
        return "none"

    _active_dispatch_mode = mode
    console_log(
        "torch trace",
        "Operator coverage: TorchDispatchMode (Python tier).",
    )
    return "torch_dispatch_mode"


def install_tensor_backward_wrapper() -> None:
    """Wrap ``torch.Tensor.backward``."""
    if getattr(torch.Tensor.backward, "_roctx_wrapped", False):
        return

    original_backward = torch.Tensor.backward
    backward_counter = {"count": 0}

    def backward_with_roctx(
        self: Any,  # noqa: ANN401
        *args: Any,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        backward_counter["count"] += 1
        location = resolve_user_caller_location()
        _push_scope("torch.Tensor.backward", f"#{backward_counter['count']}@{location}")
        try:
            return original_backward(self, *args, **kwargs)
        finally:
            _pop_scope()

    backward_with_roctx._roctx_wrapped = True
    torch.Tensor.backward = backward_with_roctx
    console_log("torch trace", "Wrapped torch.Tensor.backward with ROCTX markers")


def wrap_method_on_subclasses(
    base_class: type,
    method_name: str,
    wrapper_factory: Callable[..., Any],
) -> int:
    """Wrap ``method_name`` on every subclass of ``base_class``.

    Future subclasses are wrapped via an ``__init__`` hook. Returns the
    number of method definitions newly wrapped.
    """
    wrapped_classes = set()
    wrapped_method_count = {"count": 0}

    def wrap_class(cls: type) -> None:
        if cls in wrapped_classes:
            return
        wrapped_classes.add(cls)
        try:
            for ancestor in cls.__mro__:
                if method_name in ancestor.__dict__:
                    fn = ancestor.__dict__[method_name]
                    if not getattr(fn, "_roctx_wrapped", False):
                        wrapped_fn = wrapper_factory(fn)
                        wrapped_fn._roctx_wrapped = True
                        setattr(ancestor, method_name, wrapped_fn)
                        wrapped_method_count["count"] += 1
                    break
        except Exception as e:
            console_warning(
                "torch trace",
                f"Failed to wrap {cls.__name__}.{method_name}: {e}",
            )

    _walk_subclasses(base_class, wrap_class)
    wrap_class(base_class)

    existing_init = base_class.__init__
    if not getattr(existing_init, "_roctx_init_hook", False):
        original_init = existing_init

        def init_hook(self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            cls = type(self)
            if cls not in wrapped_classes:
                wrap_class(cls)
            return original_init(self, *args, **kwargs)

        init_hook._roctx_init_hook = True
        base_class.__init__ = init_hook

    return wrapped_method_count["count"]


def inject_roctx_into_optimizer() -> None:
    """Wrap ``step()`` on every ``torch.optim`` optimizer."""
    if Optimizer is None:
        return

    def make_step_wrapper(original_step: Callable[..., Any]) -> Callable[..., Any]:
        def step_with_roctx(
            self: Any,  # noqa: ANN401
            *args: Any,
            **kwargs: Any,
        ) -> Any:  # noqa: ANN401
            location = resolve_user_caller_location()
            if not hasattr(self, "_roctx_step_call_count"):
                self._roctx_step_call_count = 0
            self._roctx_step_call_count += 1
            _push_scope(
                f"optimizer.{type(self).__name__}.step",
                f"#{self._roctx_step_call_count}@{location}",
            )
            try:
                return original_step(self, *args, **kwargs)
            finally:
                _pop_scope()

        return step_with_roctx

    wrapped_count = wrap_method_on_subclasses(Optimizer, "step", make_step_wrapper)
    if wrapped_count > 0:
        console_log(
            "torch trace",
            "Wrapped optimizer.step() across torch.optim subclasses "
            "with ROCTX markers\n",
        )


def wrap_module_function(
    module: Any,  # noqa: ANN401
    attr_name: str,
    marker_name: str,
) -> bool:
    """Replace ``module.attr_name`` with a ROCTX-wrapped version. Never raises."""
    fn = getattr(module, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, "_roctx_wrapped", False):
        return True
    wrapped = roctx_wrapper(fn, marker_name)
    try:
        setattr(module, attr_name, wrapped)
    except Exception as e:
        console_warning(
            "torch trace",
            f"Could not patch {marker_name}: {e}",
        )
        return False
    return True


EXTRA_STRUCTURAL_WRAPS = (
    ("torch.autograd", "grad", "torch.autograd.grad"),
    ("torch.autograd.functional", "hessian", "torch.autograd.functional.hessian"),
    ("torch.autograd.functional", "jacobian", "torch.autograd.functional.jacobian"),
    ("torch.autograd.functional", "jvp", "torch.autograd.functional.jvp"),
    ("torch.autograd.functional", "vjp", "torch.autograd.functional.vjp"),
    ("torch.autograd.functional", "hvp", "torch.autograd.functional.hvp"),
    ("torch.autograd.functional", "vhp", "torch.autograd.functional.vhp"),
    ("torch.cuda", "synchronize", "torch.cuda.synchronize"),
    ("torch.cuda", "current_device", "torch.cuda.current_device"),
    ("torch.cuda", "device_count", "torch.cuda.device_count"),
    ("torch.cuda", "empty_cache", "torch.cuda.empty_cache"),
    ("torch.cuda", "manual_seed", "torch.cuda.manual_seed"),
    ("torch.cuda", "memory_allocated", "torch.cuda.memory_allocated"),
    ("torch.cuda", "reset_peak_memory_stats", "torch.cuda.reset_peak_memory_stats"),
    ("torch.cuda", "set_device", "torch.cuda.set_device"),
    ("torch.jit", "script", "torch.jit.script"),
    ("torch.jit", "trace", "torch.jit.trace"),
    ("torch", "argmax", "torch.argmax"),
    ("torch", "sum", "torch.sum"),
    ("torch", "eq", "torch.eq"),
    ("torch", "mean", "torch.mean"),
    ("torch", "max", "torch.max"),
    ("torch.nn.functional", "nll_loss", "torch.nn.functional.nll_loss"),
    ("torch.nn.functional", "cross_entropy", "torch.nn.functional.cross_entropy"),
    ("torch.nn.functional", "mse_loss", "torch.nn.functional.mse_loss"),
    ("torch.nn.functional", "log_softmax", "torch.nn.functional.log_softmax"),
    ("torch.nn.functional", "softmax", "torch.nn.functional.softmax"),
    ("torch.nn.functional", "relu", "torch.nn.functional.relu"),
)


# Tensor methods commonly used as entry points.
TENSOR_METHOD_WRAPS = (
    "item",
    "argmax",
    "sum",
    "mean",
    "max",
    "eq",
    "numpy",
    "tolist",
)

DEEP_TENSOR_METHOD_WRAPS = (
    "to",
    "cpu",
    "cuda",
    "contiguous",
)

DEEP_TENSOR_METHOD_WRAPS_ENV = "ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS"


def _deep_tensor_method_wraps_enabled() -> bool:
    value = os.environ.get(DEEP_TENSOR_METHOD_WRAPS_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")


def _selected_tensor_method_wraps() -> tuple[str, ...]:
    if _deep_tensor_method_wraps_enabled():
        return TENSOR_METHOD_WRAPS + DEEP_TENSOR_METHOD_WRAPS
    return TENSOR_METHOD_WRAPS


def install_function_apply_wrappers() -> bool:
    """Wrap ``Function.apply`` on every subclass, including future ones."""
    if Function is None:
        return False

    def stamp_apply(cls: type) -> None:
        try:
            for ancestor in cls.__mro__:
                existing = ancestor.__dict__.get("apply")
                existing_fn = getattr(existing, "__func__", existing)
                if existing is not None and getattr(
                    existing_fn, "_roctx_wrapped", False
                ):
                    return
        except Exception:
            return
        try:
            base_apply = cls.apply
        except Exception:
            return

        def wrapped_apply(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            location = resolve_user_caller_location()
            _push_scope("torch.autograd.Function.apply", f"#1@{location}")
            try:
                return base_apply(*args, **kwargs)
            finally:
                _pop_scope()

        wrapped_apply._roctx_wrapped = True
        try:
            cls.apply = staticmethod(wrapped_apply)
        except Exception:
            return

    _walk_subclasses(Function, stamp_apply)

    existing_isc = Function.__init_subclass__
    existing_isc_fn = getattr(existing_isc, "__func__", existing_isc)
    if getattr(existing_isc_fn, "_roctx_function_subclass_hook", False):
        return True

    original_init_subclass = existing_isc

    def init_subclass_hook(cls: type, **kwargs: Any) -> None:
        try:
            original_init_subclass_fn = getattr(
                original_init_subclass,
                "__func__",
                original_init_subclass,
            )
            original_init_subclass_fn(cls, **kwargs)
        except Exception:
            pass
        try:
            stamp_apply(cls)
        except Exception as e:
            console_warning(
                "torch trace",
                f"stamp_apply({cls.__name__}) failed in __init_subclass__: {e}",
            )

    init_subclass_hook._roctx_function_subclass_hook = True
    Function.__init_subclass__ = classmethod(init_subclass_hook)
    return True


def install_tensor_method_wrappers() -> None:
    """Wrap selected ``torch.Tensor`` methods.

    ``DEEP_TENSOR_METHOD_WRAPS`` are opt-in via
    ``ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=1``.
    """
    wrapped = []
    selected_methods = _selected_tensor_method_wraps()
    if not _deep_tensor_method_wraps_enabled():
        console_log(
            "torch trace",
            "Deep tensor method wraps disabled by default; set "
            f"{DEEP_TENSOR_METHOD_WRAPS_ENV}=1 to enable "
            f"({', '.join(DEEP_TENSOR_METHOD_WRAPS)}).",
        )
    for method_name in selected_methods:
        fn = getattr(torch.Tensor, method_name, None)
        if fn is None or not callable(fn):
            continue
        if getattr(fn, "_roctx_wrapped", False):
            continue
        try:
            wrapped_fn = roctx_wrapper(fn, f"torch.Tensor.{method_name}")
            setattr(torch.Tensor, method_name, wrapped_fn)
            wrapped.append(method_name)
        except (TypeError, AttributeError) as e:
            console_warning(
                "torch trace",
                f"Could not patch torch.Tensor.{method_name}: {e}",
            )

    if wrapped:
        console_log(
            "torch trace",
            f"Wrapped {len(wrapped)} torch.Tensor methods with ROCTX markers: "
            f"{', '.join(wrapped)}",
        )


def install_extra_structural_wrappers() -> None:
    """Wrap additional entry points and ``torch.cuda.{Event,Stream}``."""
    wrapped = []
    for module_path, attr_name, marker_name in EXTRA_STRUCTURAL_WRAPS:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        if wrap_module_function(module, attr_name, marker_name):
            wrapped.append(marker_name)

    install_tensor_method_wrappers()

    if cuda_mod is not None:
        for cls_name in ("Event", "Stream"):
            cls = getattr(cuda_mod, cls_name, None)
            if cls is None:
                continue
            init = getattr(cls, "__init__", None)
            if init is None or getattr(init, "_roctx_wrapped", False):
                continue
            try:
                if init is object.__init__:
                    cls.__init__ = _marker_only_init_wrapper(f"torch.cuda.{cls_name}")
                else:
                    wrapped_init = roctx_wrapper(init, f"torch.cuda.{cls_name}")
                    cls.__init__ = wrapped_init
                wrapped.append(f"torch.cuda.{cls_name}")
            except Exception as e:
                console_warning(
                    "torch trace",
                    f"Could not patch torch.cuda.{cls_name}.__init__: {e}",
                )

    try:
        if install_function_apply_wrappers():
            wrapped.append("torch.autograd.Function.apply")
    except Exception as e:
        console_warning(
            "torch trace",
            f"Could not patch torch.autograd.Function.apply: {e}",
        )

    if wrapped:
        console_log(
            "torch trace",
            f"Wrapped {len(wrapped)} additional structural entry points "
            "with ROCTX markers",
        )


def inject_roctx_into_model() -> None:
    """Wrap ``nn.Module.__call__`` so module hooks are also covered."""
    if nn is None:
        return
    if getattr(nn.Module.__call__, "_roctx_wrapped", False):
        return

    original_call = nn.Module.__call__

    def call_with_roctx(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        class_name = self.__class__.__name__
        if not hasattr(self, "_roctx_call_count"):
            self._roctx_call_count = 0
        self._roctx_call_count += 1
        location = resolve_user_caller_location()
        _push_scope(
            f"nn.Module.{class_name}.forward",
            f"#{self._roctx_call_count}@{location}",
        )
        try:
            return original_call(self, *args, **kwargs)
        finally:
            _pop_scope()

    call_with_roctx._roctx_wrapped = True
    did_wrap = False
    try:
        nn.Module.__call__ = call_with_roctx
        did_wrap = nn.Module.__call__ is call_with_roctx
    except Exception as e:
        console_warning("torch trace", f"Could not patch nn.Module.__call__: {e}")
    if did_wrap:
        console_log("torch trace", "Wrapped nn.Module forward() with ROCTX markers\n")


def using_c_tier() -> bool:
    """Return True when the C++ RecordFunction tier is active."""
    return _USING_C_TIER


def dump_recordfn_stats() -> Optional[dict[str, Any]]:  # noqa: ANN401
    """Return the C++ extension's counters, or ``None`` on the Python tier."""
    if _roctx_recordfn is None:
        return None
    try:
        return _roctx_recordfn.dump_stats()
    except Exception:
        return None


def install_global_wraps() -> None:
    """Apply every process-global patch. Idempotent."""
    _initialize_c_tier()
    _emit_python_tier_fallback_warning()
    install_dispatcher_hook()
    patch_distributed_collectives()
    patch_process_group_methods()
    patch_cuda_graph()
    patch_triton_launcher()
    patch_compile_callable()
    install_tensor_backward_wrapper()
    install_extra_structural_wrappers()
    inject_roctx_into_optimizer()
    inject_roctx_into_model()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        console_log(
            "torch trace", "Usage: python inject_roctx.py <script.py> [script_args...]"
        )
        sys.exit(1)

    target_script = sys.argv[1]
    script_args = sys.argv[2:]

    install_global_wraps()

    console_log("=" * 70)
    console_log("torch trace", "Starting target script with ROCTX instrumentation...")
    console_log("=" * 70)

    sys.argv = [target_script] + script_args
    spec = importlib.util.spec_from_file_location("__main__", target_script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        # Surface any callback errors swallowed by the C++ tier.
        _stats = dump_recordfn_stats()
        if _stats is not None:
            _cb_errors = int(_stats.get("callback_errors", 0) or 0)
            if _cb_errors > 0:
                console_warning(
                    "torch trace",
                    f"roctx_recordfn observed {_cb_errors} swallowed "
                    "callback exception(s) during the workload. Some "
                    "ROCTX markers may be missing or misattributed. "
                    f"Stats: {dict(_stats)}",
                )
