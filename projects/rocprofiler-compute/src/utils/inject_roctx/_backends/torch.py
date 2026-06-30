# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""ROCTX instrumentation backend for PyTorch.

ATen ops use the C++ RecordFunction tier (roctx_recordfn.so) when
TorchBackend.install() initializes it, with a Python TorchDispatchMode
fallback. Structural wraps (nn.Module, Optimizer, distributed,
CUDA graph, torch.compile) run on both tiers. Triton kernel launches
are wrapped by the separate triton backend.

Module import is side-effect free with respect to PyTorch state. Torch is
imported lazily by TorchBackend.install.
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

from utils.inject_roctx import core
from utils.inject_roctx.registry import register
from utils.logger import console_log, console_warning

_BACKEND_NAME = "torch"


class _RecordFnHook:
    def active(self) -> bool:
        return _STATE.using_c_tier and _STATE.roctx_recordfn is not None

    def push(self, marker: str, context: str, backend: str) -> bool:
        try:
            _STATE.roctx_recordfn.push_user_scope(marker, context, backend)
            return True
        except Exception:
            return False

    def pop(self) -> None:
        try:
            _STATE.roctx_recordfn.pop_user_scope()
        except Exception:
            pass


class _TorchState:
    """Mutable backend state populated by TorchBackend.install().

    Holds the resolved torch module handles, the roctx_recordfn loader and
    native C++ RecordFunction tier, and the active Python-tier dispatch mode.
    """

    def __init__(self) -> None:
        self.torch: Any = None
        self.dist: Any = None
        self.fc: Any = None
        self.process_group: Any = None
        self.cuda_mod: Any = None
        self.torch_dispatch_mode: Any = None
        self.optimizer: Any = None
        self.function: Any = None
        self.nn: Any = None
        self.torch_root: str = ""

        self.loader_module: Any = None
        self.load_roctx_recordfn: Optional[Callable[..., Any]] = None
        self.load_diagnostics: list[tuple[str, str]] = []
        self.roctx_recordfn: Any = None
        self.using_c_tier: bool = False
        self.c_tier_initialized: bool = False
        self.native_hook: Optional[_RecordFnHook] = None

        self.active_dispatch_mode: Any = None


_STATE = _TorchState()

_thread_local = threading.local()

rangePush: Optional[Callable[[str], None]] = None
rangePop: Optional[Callable[[], None]] = None

# Wire the Python tier via core and reuse its roctx handles below.
_ROCTX_AVAILABLE = core.ensure_python_tier()
if _ROCTX_AVAILABLE:
    rangePush, rangePop = core.get_python_tier_io()


# torch.distributed.* collectives; entries not listed here are not wrapped.
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

# ProcessGroup methods FSDP2/DTensor call directly. Wrapped per subclass.
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


# The native C++ RecordFunction tier is installed by _initialize_c_tier().


def _get_tier_stack() -> list[bool]:
    # Per-frame record of the tier that handled each push: True for the native
    # C++ RecordFunction tier, False for the Python tier.
    if not hasattr(_thread_local, "tier_stack"):
        _thread_local.tier_stack = []
    return _thread_local.tier_stack


def _push_scope(marker: str, context: str, backend: str = "") -> None:
    """Push a scope, routing through the native C++ RecordFunction tier when
    active and otherwise emitting on the Python tier.
    """
    marker_stack = core.get_marker_stack()
    context_stack = core.get_context_stack()
    tier_stack = _get_tier_stack()

    used_native = False
    hook = _STATE.native_hook
    if hook is not None and hook.active():
        try:
            used_native = bool(hook.push(marker, context, backend))
        except Exception:
            used_native = False

    if not used_native:
        full = core.compose_marker(marker, context, backend)
        range_push, _ = core.get_python_tier_io()
        range_push(full)

    tier_stack.append(used_native)
    marker_stack.append(marker)
    context_stack.append(context)


def _pop_scope() -> None:
    """Pop a scope, routing to the tier that handled its push."""
    marker_stack = core.get_marker_stack()
    context_stack = core.get_context_stack()
    tier_stack = _get_tier_stack()

    # Unmatched pop: no-op.
    if not tier_stack:
        return

    used_native = tier_stack.pop()
    try:
        if used_native and _STATE.native_hook is not None:
            _STATE.native_hook.pop()
        else:
            _, range_pop = core.get_python_tier_io()
            range_pop()
    finally:
        if marker_stack:
            marker_stack.pop()
        if context_stack:
            context_stack.pop()


# Structural wrappers for entry points the ATen dispatcher does not record.


def roctx_wrapper(
    func: Callable[..., Any],
    name: Optional[str] = None,
    backend: str = "",
) -> Callable[..., Any]:
    """Wrap func so each call emits a ROCTX range. Idempotent.

    A non-empty backend attributes the scope to that backend.
    """
    if getattr(func, "_roctx_wrapped", False):
        return func
    func_name = name or func.__name__
    call_counter = {"count": 0}

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> object:
        call_counter["count"] += 1
        location = core.resolve_user_caller_location()
        _push_scope(func_name, f"#{call_counter['count']}@{location}", backend=backend)
        try:
            return func(*args, **kwargs)
        finally:
            _pop_scope()

    wrapper._roctx_wrapped = True
    return wrapper


def _marker_only_init_wrapper(name: str, backend: str = "") -> Callable[..., Any]:
    """Build an __init__ that emits a ROCTX range, then calls object.__init__.

    Used for classes whose construction occurs in __new__ (e.g. cuda.Event,
    cuda.Stream).
    """
    call_counter = {"count": 0}

    def marker_only_init(self: object, *args: Any, **kwargs: Any) -> None:
        call_counter["count"] += 1
        location = core.resolve_user_caller_location()
        _push_scope(name, f"#{call_counter['count']}@{location}", backend=backend)
        try:
            return object.__init__(self)
        finally:
            _pop_scope()

    marker_only_init._roctx_wrapped = True
    return marker_only_init


def _walk_subclasses(cls: type, fn: Callable[[type], None]) -> None:
    """Apply `fn` to every (transitive) subclass of `cls`."""
    for sub in cls.__subclasses__():
        fn(sub)
        _walk_subclasses(sub, fn)


def _initialize_c_tier() -> bool:
    """Load and install roctx_recordfn.so once per process."""
    if _STATE.c_tier_initialized:
        return _STATE.using_c_tier

    _STATE.c_tier_initialized = True

    if _STATE.load_roctx_recordfn is None:
        _STATE.roctx_recordfn = None
        _STATE.using_c_tier = False
        return False

    try:
        result = _STATE.load_roctx_recordfn()
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"loader raised; falling back to Python tier: {exc}",
        )
        result = None

    if result is not None:
        _STATE.roctx_recordfn = result.module
        _STATE.load_diagnostics = result.diagnostics
    else:
        _STATE.roctx_recordfn = None

    if _STATE.roctx_recordfn is not None:
        try:
            _STATE.roctx_recordfn.install()
            console_log(
                "ml api trace",
                (
                    "Coverage tier: C++ RecordFunction "
                    "(global callback; covers every thread)."
                ),
            )
            _STATE.using_c_tier = True
            _STATE.native_hook = _RecordFnHook()
            return True
        except Exception as exc:
            console_warning(
                "ml api trace",
                f".so install() raised; falling back to Python tier: {exc}",
            )
            _STATE.roctx_recordfn = None

    _STATE.using_c_tier = False
    return False


def _emit_python_tier_fallback_warning() -> None:
    """Emit one warning when the C++ tier is unavailable."""
    if _STATE.using_c_tier:
        return
    loader_trail = ""
    if _STATE.loader_module is not None:
        try:
            loader_trail = _STATE.loader_module.format_load_diagnostic_trail(
                _STATE.load_diagnostics,
            )
        except Exception:
            loader_trail = ""
    trail_block = f"\nLoader trail:\n{loader_trail}" if loader_trail else ""
    console_warning(
        "ml api trace",
        "Coverage tier: Python-only injector (the C++ RecordFunction tier is "
        "unavailable). Operator coverage on autograd backward threads is "
        "reduced; backward markers may lack their full context chain." + trail_block,
    )


def patch_distributed_collectives() -> None:
    """Wrap DISTRIBUTED_COLLECTIVE_NAMES + every entry in _functional_collectives."""
    dist = _STATE.dist
    if dist is None:
        console_warning(
            "ml api trace",
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
            setattr(
                dist,
                fn_name,
                roctx_wrapper(
                    fn,
                    f"torch.distributed.{fn_name}",
                    backend=_BACKEND_NAME,
                ),
            )
            wrapped.append(fn_name)
        except Exception as exc:
            console_warning(
                "ml api trace",
                f"Could not patch torch.distributed.{fn_name}: {exc}",
            )

    fc = _STATE.fc
    if fc is not None:
        for fn_name in dir(fc):
            if fn_name.startswith("_"):
                continue
            fn = getattr(fc, fn_name, None)
            if not callable(fn):
                continue
            if inspect.isclass(fn):
                continue
            # Skip symbols re-exported from other modules (e.g. ReduceOp).
            if getattr(fn, "__module__", "") != fc.__name__:
                continue
            try:
                setattr(
                    fc,
                    fn_name,
                    roctx_wrapper(
                        fn,
                        f"torch.distributed._functional_collectives.{fn_name}",
                        backend=_BACKEND_NAME,
                    ),
                )
                wrapped.append(f"_functional_collectives.{fn_name}")
            except Exception as exc:
                console_warning(
                    "ml api trace",
                    f"Could not patch _functional_collectives.{fn_name}: {exc}",
                )

    if wrapped:
        console_log(
            "ml api trace",
            f"Wrapped {len(wrapped)} torch.distributed collectives with ROCTX markers",
        )


def patch_process_group_methods() -> None:
    """Wrap ProcessGroup methods on every existing subclass.

    An __init_subclass__ hook covers subclasses registered later.
    """
    process_group = _STATE.process_group
    if process_group is None:
        return

    wrapped_classes: set[type] = set()
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
                wrapped = roctx_wrapper(
                    fn,
                    marker,
                    backend=_BACKEND_NAME,
                )
                setattr(cls, method_name, wrapped)
                wrapped_method_count["count"] += 1
            except Exception as exc:
                console_warning(
                    "ml api trace",
                    f"Could not patch ProcessGroup.{cls.__name__}.{method_name}: {exc}",
                )

    _wrap_one(process_group)
    _walk_subclasses(process_group, _wrap_one)

    existing_isc = process_group.__init_subclass__
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
            except Exception as exc:
                console_warning(
                    "ml api trace",
                    f"_wrap_one({cls.__name__}) failed in "
                    f"ProcessGroup.__init_subclass__: {exc}",
                )

        init_subclass_hook._roctx_pg_subclass_hook = True
        try:
            process_group.__init_subclass__ = classmethod(init_subclass_hook)
        except Exception:
            # C-defined on some builds; per-subclass walk above covers existing.
            pass

    if wrapped_method_count["count"]:
        console_log(
            "ml api trace",
            f"Wrapped {wrapped_method_count['count']} ProcessGroup methods across "
            f"{len(wrapped_classes)} subclasses with ROCTX markers",
        )


def patch_cuda_graph() -> None:
    """Wrap CUDAGraph.capture_begin, capture_end, and replay.

    A replay runs as a single hipGraphLaunch with no per-op records.
    """
    cuda_mod = _STATE.cuda_mod
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
            wrapped = roctx_wrapper(
                fn,
                marker,
                backend=_BACKEND_NAME,
            )
            setattr(cls, method_name, wrapped)
            wrapped_methods.append(method_name)
        except Exception as exc:
            console_warning(
                "ml api trace",
                f"Could not patch CUDAGraph.{method_name}: {exc}",
            )

    if wrapped_methods:
        console_log(
            "ml api trace",
            "Wrapped CUDAGraph methods with ROCTX markers: "
            f"{', '.join(wrapped_methods)}",
        )


def patch_compile_callable() -> None:
    """Wrap torch.compile and the returned callable so each invocation is bracketed."""
    torch = _STATE.torch
    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        return
    if getattr(original_compile, "_roctx_wrapped", False):
        return

    @wraps(original_compile)
    def compile_with_roctx(
        model_or_fn: object = None,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        location = core.resolve_user_caller_location()
        _push_scope("torch.compile", f"#1@{location}", backend=_BACKEND_NAME)
        try:
            compiled = original_compile(model_or_fn, *args, **kwargs)
        finally:
            _pop_scope()

        if compiled is None or not callable(compiled):
            return compiled

        fn_label = getattr(model_or_fn, "__name__", None) or type(model_or_fn).__name__

        @wraps(compiled)
        def invocation_wrapper(*c_args: Any, **c_kwargs: Any) -> object:
            loc = core.resolve_user_caller_location()
            _push_scope(f"torch.compile.{fn_label}", f"#1@{loc}", backend=_BACKEND_NAME)
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
            "ml api trace",
            "Wrapped torch.compile + its returned callable with ROCTX markers",
        )
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch torch.compile invocation wrapper: {exc}",
        )


# Dispatcher: C++ tier covers fwd+bwd; Python tier covers forward only.


def next_dispatcher_index(op_name: str) -> int:
    """Per-thread occurrence count for op_name."""
    counters = getattr(_thread_local, "dispatcher_counters", None)
    if counters is None:
        counters = {}
        _thread_local.dispatcher_counters = counters
    counters[op_name] = counters.get(op_name, 0) + 1
    return counters[op_name]


def warn_dispatcher_failure_once(phase: str, error: Exception) -> None:
    """Emit one warning per (thread, phase)."""
    flag_attr = f"warned_dispatcher_failure_{phase}"
    if getattr(_thread_local, flag_attr, False):
        return
    setattr(_thread_local, flag_attr, True)
    try:
        console_warning(
            "ml api trace",
            f"Dispatcher {phase} raised ({type(error).__name__}: {error}). "
            "Subsequent failures on this thread will be suppressed.",
        )
    except Exception:
        pass


def dispatcher_marker_name_for(func: Callable[..., Any]) -> str:
    """Map ATen overloads to torch.ops.aten.<op>; other namespaces to <ns>::<op>."""
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
    """C++ tier: no-op. Python tier: enter TorchDispatchMode on this thread."""
    if _STATE.using_c_tier:
        console_log(
            "ml api trace",
            "Operator coverage: C++ RecordFunction callback "
            "(FUNCTION + BACKWARD_FUNCTION).",
        )
        return "c_tier"

    torch_dispatch_mode = _STATE.torch_dispatch_mode
    if torch_dispatch_mode is None:
        console_warning(
            "ml api trace",
            "TorchDispatchMode is not importable on this PyTorch build; "
            "per-op coverage will be missing.",
        )
        return "none"

    def start_disp(op_name: str) -> None:
        idx = next_dispatcher_index(op_name)
        location = core.resolve_user_caller_location()
        marker_stack = core.get_marker_stack()
        context_stack = core.get_context_stack()
        context = f"#{idx}@{location}"
        rangePush(core.compose_marker(op_name, context, _BACKEND_NAME))
        marker_stack.append(op_name)
        context_stack.append(context)

    def end_disp() -> None:
        marker_stack = core.get_marker_stack()
        context_stack = core.get_context_stack()
        try:
            rangePop()
        finally:
            if marker_stack:
                marker_stack.pop()
            if context_stack:
                context_stack.pop()

    class RoctxDispatchMode(torch_dispatch_mode):
        def __torch_dispatch__(
            self,
            func: Callable[..., Any],
            types: tuple[type, ...],
            args: tuple[object, ...] = (),
            kwargs: Optional[dict[str, object]] = None,
        ) -> object:
            kwargs = kwargs or {}
            op_name = dispatcher_marker_name_for(func)
            pushed = False
            try:
                start_disp(op_name)
                pushed = True
            except Exception as exc:
                warn_dispatcher_failure_once("start", exc)
            try:
                return func(*args, **kwargs)
            finally:
                if pushed:
                    try:
                        end_disp()
                    except Exception as exc:
                        warn_dispatcher_failure_once("end", exc)

    try:
        mode = RoctxDispatchMode()
        mode.__enter__()
    except Exception as exc:
        console_warning("ml api trace", f"TorchDispatchMode activation failed: {exc}")
        return "none"

    _STATE.active_dispatch_mode = mode
    console_log(
        "ml api trace",
        "Operator coverage: TorchDispatchMode (Python tier).",
    )
    return "torch_dispatch_mode"


def install_tensor_backward_wrapper() -> None:
    """Wrap torch.Tensor.backward; per-op backward dispatches go to the tier."""
    torch = _STATE.torch
    if getattr(torch.Tensor.backward, "_roctx_wrapped", False):
        return

    original_backward = torch.Tensor.backward
    backward_counter = {"count": 0}

    def backward_with_roctx(
        self: object,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        backward_counter["count"] += 1
        location = core.resolve_user_caller_location()
        _push_scope(
            "torch.Tensor.backward",
            f"#{backward_counter['count']}@{location}",
            backend=_BACKEND_NAME,
        )
        try:
            return original_backward(self, *args, **kwargs)
        finally:
            _pop_scope()

    backward_with_roctx._roctx_wrapped = True
    torch.Tensor.backward = backward_with_roctx
    console_log("ml api trace", "Wrapped torch.Tensor.backward with ROCTX markers")


def wrap_method_on_subclasses(
    base_class: type,
    method_name: str,
    wrapper_factory: Callable[..., Any],
) -> int:
    """Wrap method_name on every defining class; future subclasses via __init__ hook.

    Returns the count of method definitions newly wrapped.
    """
    wrapped_classes: set[type] = set()
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
        except Exception as exc:
            console_warning(
                "ml api trace",
                f"Failed to wrap {cls.__name__}.{method_name}: {exc}",
            )

    _walk_subclasses(base_class, wrap_class)
    wrap_class(base_class)

    existing_init = base_class.__init__
    if not getattr(existing_init, "_roctx_init_hook", False):
        original_init = existing_init

        def init_hook(self: object, *args: Any, **kwargs: Any) -> None:
            cls = type(self)
            if cls not in wrapped_classes:
                wrap_class(cls)
            return original_init(self, *args, **kwargs)

        init_hook._roctx_init_hook = True
        base_class.__init__ = init_hook

    return wrapped_method_count["count"]


def inject_roctx_into_optimizer() -> None:
    """Wrap step() on every torch.optim optimizer."""
    optimizer = _STATE.optimizer
    if optimizer is None:
        return

    def make_step_wrapper(original_step: Callable[..., Any]) -> Callable[..., Any]:
        def step_with_roctx(
            self: object,
            *args: Any,
            **kwargs: Any,
        ) -> object:
            location = core.resolve_user_caller_location()
            if not hasattr(self, "_roctx_step_call_count"):
                self._roctx_step_call_count = 0
            self._roctx_step_call_count += 1
            _push_scope(
                f"optimizer.{type(self).__name__}.step",
                f"#{self._roctx_step_call_count}@{location}",
                backend=_BACKEND_NAME,
            )
            try:
                return original_step(self, *args, **kwargs)
            finally:
                _pop_scope()

        return step_with_roctx

    wrapped_count = wrap_method_on_subclasses(optimizer, "step", make_step_wrapper)
    if wrapped_count > 0:
        console_log(
            "ml api trace",
            "Wrapped optimizer.step() across torch.optim subclasses "
            "with ROCTX markers\n",
        )


def wrap_module_function(
    module: object,
    attr_name: str,
    marker_name: str,
) -> bool:
    """Replace module.attr_name with a ROCTX-wrapped version. Never raises."""
    fn = getattr(module, attr_name, None)
    if fn is None or not callable(fn):
        return False
    if getattr(fn, "_roctx_wrapped", False):
        return True
    wrapped = roctx_wrapper(
        fn,
        marker_name,
        backend=_BACKEND_NAME,
    )
    try:
        setattr(module, attr_name, wrapped)
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch {marker_name}: {exc}",
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
    # Top-level ATen wrappers also reachable as Tensor methods; both routes bracketed.
    ("torch", "argmax", "torch.argmax"),
    ("torch", "sum", "torch.sum"),
    ("torch", "eq", "torch.eq"),
    ("torch", "mean", "torch.mean"),
    ("torch", "max", "torch.max"),
    # Functional losses: outer Python facade gets user-frame attribution.
    ("torch.nn.functional", "nll_loss", "torch.nn.functional.nll_loss"),
    ("torch.nn.functional", "cross_entropy", "torch.nn.functional.cross_entropy"),
    ("torch.nn.functional", "mse_loss", "torch.nn.functional.mse_loss"),
    ("torch.nn.functional", "log_softmax", "torch.nn.functional.log_softmax"),
    ("torch.nn.functional", "softmax", "torch.nn.functional.softmax"),
    ("torch.nn.functional", "relu", "torch.nn.functional.relu"),
)


# Tensor methods often used as entry points (e.g. output.argmax(dim=1)).
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
    return value not in ("0", "false", "no", "off")


def _selected_tensor_method_wraps() -> tuple[str, ...]:
    if _deep_tensor_method_wraps_enabled():
        return TENSOR_METHOD_WRAPS + DEEP_TENSOR_METHOD_WRAPS
    return TENSOR_METHOD_WRAPS


def install_function_apply_wrappers() -> bool:
    """Wrap Function.apply on every existing subclass.

    An __init_subclass__ hook wraps subclasses registered later.
    """
    function = _STATE.function
    if function is None:
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

        def wrapped_apply(*args: Any, **kwargs: Any) -> object:
            location = core.resolve_user_caller_location()
            _push_scope(
                "torch.autograd.Function.apply",
                f"#1@{location}",
                backend=_BACKEND_NAME,
            )
            try:
                return base_apply(*args, **kwargs)
            finally:
                _pop_scope()

        wrapped_apply._roctx_wrapped = True
        try:
            cls.apply = staticmethod(wrapped_apply)
        except Exception:
            return

    _walk_subclasses(function, stamp_apply)

    existing_isc = function.__init_subclass__
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
        except Exception as exc:
            console_warning(
                "ml api trace",
                f"stamp_apply({cls.__name__}) failed in __init_subclass__: {exc}",
            )

    init_subclass_hook._roctx_function_subclass_hook = True
    function.__init_subclass__ = classmethod(init_subclass_hook)
    return True


def install_tensor_method_wrappers() -> None:
    """Wrap selected Tensor methods so inner ATen dispatches inherit the user frame.

    DEEP_TENSOR_METHOD_WRAPS are enabled by default; set
    ROCPROFCOMPUTE_ROCTX_DEEP_TENSOR_WRAPS=0 to disable them.
    """
    torch = _STATE.torch
    wrapped = []
    selected_methods = _selected_tensor_method_wraps()
    if not _deep_tensor_method_wraps_enabled():
        console_log(
            "ml api trace",
            f"Deep tensor method wraps disabled via {DEEP_TENSOR_METHOD_WRAPS_ENV}; "
            f"unset or set to 1 to enable ({', '.join(DEEP_TENSOR_METHOD_WRAPS)}).",
        )
    for method_name in selected_methods:
        fn = getattr(torch.Tensor, method_name, None)
        if fn is None or not callable(fn):
            continue
        if getattr(fn, "_roctx_wrapped", False):
            continue
        try:
            wrapped_fn = roctx_wrapper(
                fn,
                f"torch.Tensor.{method_name}",
                backend=_BACKEND_NAME,
            )
            setattr(torch.Tensor, method_name, wrapped_fn)
            wrapped.append(method_name)
        except (TypeError, AttributeError) as exc:
            # C-slot methods refuse Python reassignment.
            console_warning(
                "ml api trace",
                f"Could not patch torch.Tensor.{method_name}: {exc}",
            )

    if wrapped:
        console_log(
            "ml api trace",
            f"Wrapped {len(wrapped)} torch.Tensor methods with ROCTX markers: "
            f"{', '.join(wrapped)}",
        )


def install_extra_structural_wrappers() -> None:
    """Wrap EXTRA_STRUCTURAL_WRAPS, tensor methods, cuda.{Event,Stream}."""
    wrapped = []
    for module_path, attr_name, marker_name in EXTRA_STRUCTURAL_WRAPS:
        try:
            module = importlib.import_module(module_path)
        except Exception:
            continue
        if wrap_module_function(module, attr_name, marker_name):
            wrapped.append(marker_name)

    install_tensor_method_wrappers()

    cuda_mod = _STATE.cuda_mod
    if cuda_mod is not None:
        for cls_name in ("Event", "Stream"):
            cls = getattr(cuda_mod, cls_name, None)
            if cls is None:
                continue
            init = getattr(cls, "__init__", None)
            if init is None or getattr(init, "_roctx_wrapped", False):
                continue
            try:
                # cuda.{Event,Stream} construct in __new__; __init__ is
                # inherited from object.
                if init is object.__init__:
                    cls.__init__ = _marker_only_init_wrapper(
                        f"torch.cuda.{cls_name}",
                        backend=_BACKEND_NAME,
                    )
                else:
                    wrapped_init = roctx_wrapper(
                        init,
                        f"torch.cuda.{cls_name}",
                        backend=_BACKEND_NAME,
                    )
                    cls.__init__ = wrapped_init
                wrapped.append(f"torch.cuda.{cls_name}")
            except Exception as exc:
                console_warning(
                    "ml api trace",
                    f"Could not patch torch.cuda.{cls_name}.__init__: {exc}",
                )

    try:
        if install_function_apply_wrappers():
            wrapped.append("torch.autograd.Function.apply")
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"Could not patch torch.autograd.Function.apply: {exc}",
        )

    if wrapped:
        console_log(
            "ml api trace",
            f"Wrapped {len(wrapped)} additional structural entry points "
            "with ROCTX markers",
        )


def inject_roctx_into_model() -> None:
    """Wrap nn.Module.__call__ (not forward(), so hooks are covered)."""
    nn = _STATE.nn
    if nn is None:
        return
    if getattr(nn.Module.__call__, "_roctx_wrapped", False):
        return

    original_call = nn.Module.__call__

    def call_with_roctx(self: object, *args: Any, **kwargs: Any) -> object:
        class_name = self.__class__.__name__
        if not hasattr(self, "_roctx_call_count"):
            self._roctx_call_count = 0
        self._roctx_call_count += 1
        location = core.resolve_user_caller_location()
        _push_scope(
            f"nn.Module.{class_name}.forward",
            f"#{self._roctx_call_count}@{location}",
            backend=_BACKEND_NAME,
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
    except Exception as exc:
        console_warning("ml api trace", f"Could not patch nn.Module.__call__: {exc}")
    if did_wrap:
        console_log("ml api trace", "Wrapped nn.Module forward() with ROCTX markers\n")


def using_c_tier() -> bool:
    """Return True if the C++ RecordFunction tier is active."""
    return _STATE.using_c_tier


def dump_recordfn_stats() -> Optional[dict[str, object]]:
    """Return the .so counters, or None on the Python tier."""
    if _STATE.roctx_recordfn is None:
        return None
    try:
        return _STATE.roctx_recordfn.dump_stats()
    except Exception:
        return None


def _resolve_torch() -> bool:
    """Bind the torch handles on _STATE. Returns True if torch is importable."""
    if importlib.util.find_spec("torch._C") is None:
        console_warning(
            "ml api trace",
            "PyTorch is not installed or not properly configured; "
            "skipping torch instrumentation.",
        )
        return False
    try:
        import torch as _torch_mod
    except ImportError:
        console_warning(
            "ml api trace",
            "PyTorch is not installed or not properly configured; "
            "skipping torch instrumentation.",
        )
        return False

    _STATE.torch = _torch_mod
    console_log("ml api trace", f"PyTorch version: {_torch_mod.__version__}")
    try:
        _STATE.torch_root = str(Path(_torch_mod.__file__).resolve().parent) + os.sep
    except Exception:
        _STATE.torch_root = ""
    core.add_framework_root(_STATE.torch_root)

    try:
        import torch.distributed as _dist_mod

        _STATE.dist = _dist_mod
    except Exception:
        _STATE.dist = None
    try:
        import torch.distributed._functional_collectives as _fc_mod

        _STATE.fc = _fc_mod
    except Exception:
        _STATE.fc = None
    try:
        from torch.distributed.distributed_c10d import ProcessGroup as _PG

        _STATE.process_group = _PG
    except Exception:
        _STATE.process_group = None
    try:
        import torch.cuda as _cuda_mod

        _STATE.cuda_mod = _cuda_mod
    except Exception:
        _STATE.cuda_mod = None
    try:
        from torch.utils._python_dispatch import TorchDispatchMode as _TDM

        _STATE.torch_dispatch_mode = _TDM
    except Exception:
        _STATE.torch_dispatch_mode = None
    try:
        from torch.optim import Optimizer as _Opt

        _STATE.optimizer = _Opt
    except Exception:
        _STATE.optimizer = None
    try:
        from torch.autograd import Function as _Fn

        _STATE.function = _Fn
    except Exception:
        _STATE.function = None
    try:
        from torch import nn as _nn_mod

        _STATE.nn = _nn_mod
    except Exception:
        _STATE.nn = None
    try:
        from . import torch_cpp_loader as _loader_mod
        from .torch_cpp_loader import load as _loader_fn

        _STATE.loader_module = _loader_mod
        _STATE.load_roctx_recordfn = _loader_fn
    except Exception as exc:
        console_warning(
            "ml api trace",
            f"torch_cpp_loader unavailable; falling back to Python tier: {exc}",
        )

    return True


class TorchBackend:
    name = "torch"

    def install(self) -> None:
        py = f"python{sys.version_info.major}.{sys.version_info.minor}"
        console_log("ml api trace", f"Workload Python Version: {py}")

        if not _ROCTX_AVAILABLE:
            console_warning(
                "ml api trace",
                f"ROCTX bindings not found in {core.roctx_candidate_paths()}; "
                "skipping torch instrumentation.",
            )
            return

        if rangePush is not None and hasattr(rangePush, "__code__"):
            roctx_path = Path(rangePush.__code__.co_filename).parent
        else:
            roctx_path = "<unknown>"
        console_log("ml api trace", f"ROCTX module loaded from: {roctx_path}")

        if not _resolve_torch():
            return

        _initialize_c_tier()
        _emit_python_tier_fallback_warning()
        patch_distributed_collectives()
        patch_process_group_methods()
        patch_cuda_graph()
        patch_compile_callable()
        install_dispatcher_hook()
        install_tensor_backward_wrapper()
        inject_roctx_into_optimizer()
        install_function_apply_wrappers()
        install_tensor_method_wrappers()
        install_extra_structural_wrappers()
        inject_roctx_into_model()


register(TorchBackend())
