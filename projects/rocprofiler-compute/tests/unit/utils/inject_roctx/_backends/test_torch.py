# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Focused unit tests for Torch backend installation behavior."""

from collections import Counter
from types import SimpleNamespace
from typing import List

import pytest

from utils.inject_roctx._backends import torch as torch_backend


def test_roctx_wrapper_balances_launcher_tid_when_wrapped_call_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def failing_operation() -> None:
        calls.append("operation")
        raise RuntimeError("operation failed")

    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda *_args, **_kwargs: calls.append("range-push"),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: calls.append("range-pop"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: calls.append("launcher-push") or True,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: calls.append("launcher-pop") or True,
    )

    wrapped = torch_backend.roctx_wrapper(
        failing_operation,
        publish_launcher_tid=True,
    )
    with pytest.raises(RuntimeError, match="operation failed"):
        wrapped()

    assert calls == [
        "range-push",
        "launcher-push",
        "operation",
        "launcher-pop",
        "range-pop",
    ]


def test_roctx_wrapper_does_not_pop_when_launcher_tid_push_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []
    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda *_args, **_kwargs: calls.append("range-push"),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: calls.append("range-pop"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: False,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: calls.append("launcher-pop") or True,
    )

    wrapped = torch_backend.roctx_wrapper(
        lambda: "result",
        publish_launcher_tid=True,
    )

    assert wrapped() == "result"
    assert calls == ["range-push", "range-pop"]


def test_roctx_wrapper_closes_range_when_launcher_tid_push_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def raise_interrupt() -> bool:
        calls.append("launcher-push")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda *_args, **_kwargs: calls.append("range-push"),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: calls.append("range-pop"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        raise_interrupt,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: pytest.fail("an unsuccessful push must not be popped"),
    )

    wrapped = torch_backend.roctx_wrapper(
        lambda: pytest.fail("the wrapped call must not run"),
        publish_launcher_tid=True,
    )
    with pytest.raises(KeyboardInterrupt):
        wrapped()

    assert calls == ["range-push", "launcher-push", "range-pop"]


def test_roctx_wrapper_closes_range_when_launcher_tid_pop_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    def raise_interrupt() -> bool:
        calls.append("launcher-pop")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda *_args, **_kwargs: calls.append("range-push"),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: calls.append("range-pop"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: calls.append("launcher-push") or True,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        raise_interrupt,
    )

    wrapped = torch_backend.roctx_wrapper(
        lambda: calls.append("operation"),
        publish_launcher_tid=True,
    )
    with pytest.raises(KeyboardInterrupt):
        wrapped()

    assert calls == [
        "range-push",
        "launcher-push",
        "operation",
        "launcher-pop",
        "range-pop",
    ]


def test_roctx_wrapper_warns_once_when_launcher_tid_pop_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []
    warnings: List[str] = []
    monkeypatch.setattr(
        torch_backend._thread_local,
        "warned_launcher_tid_pop_failure",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda *_args, **_kwargs: calls.append("range-push"),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: calls.append("range-pop"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: True,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: calls.append("launcher-pop") or False,
    )
    monkeypatch.setattr(
        torch_backend,
        "console_warning",
        lambda _category, message: warnings.append(message),
    )

    wrapped = torch_backend.roctx_wrapper(
        lambda: "result",
        publish_launcher_tid=True,
    )

    assert wrapped() == "result"
    assert wrapped() == "result"
    assert calls == [
        "range-push",
        "launcher-pop",
        "range-pop",
        "range-push",
        "launcher-pop",
        "range-pop",
    ]
    assert len(warnings) == 1
    assert "stale launcher ID" in warnings[0]


def test_tensor_backward_wrapper_publishes_launcher_tid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: List[str] = []

    class FakeTensor:
        def backward(self) -> str:
            calls.append("backward")
            return "result"

    monkeypatch.setattr(
        torch_backend._STATE, "torch", SimpleNamespace(Tensor=FakeTensor)
    )
    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(torch_backend, "_push_scope", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: None)
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: calls.append("launcher-push") or True,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: calls.append("launcher-pop") or True,
    )

    torch_backend.install_tensor_backward_wrapper()

    assert FakeTensor().backward() == "result"
    assert calls == ["launcher-push", "backward", "launcher-pop"]


def test_process_group_subclass_preserves_original_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingProcessGroup:
        @classmethod
        def __init_subclass__(cls, **kwargs: object) -> None:
            raise RuntimeError("invalid process group subclass")

    monkeypatch.setattr(torch_backend._STATE, "process_group", RejectingProcessGroup)
    torch_backend.patch_process_group_methods()

    with pytest.raises(RuntimeError, match="invalid process group subclass"):

        class InvalidProcessGroup(RejectingProcessGroup):
            pass


def test_process_group_subclass_forwards_kwargs_and_wraps_future_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: List[str] = []

    class PlainProcessGroup:
        @classmethod
        def __init_subclass__(cls, registration: str = "", **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            forwarded.append(registration)

    monkeypatch.setattr(torch_backend._STATE, "process_group", PlainProcessGroup)
    torch_backend.patch_process_group_methods()

    class ValidProcessGroup(PlainProcessGroup, registration="forwarded"):
        def allreduce(self) -> str:
            return "allreduce"

    assert forwarded == ["forwarded"]
    assert getattr(ValidProcessGroup.allreduce, "_roctx_wrapped", False)


def test_autograd_function_subclass_preserves_original_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RejectingFunction:
        @classmethod
        def __init_subclass__(cls, **kwargs: object) -> None:
            raise RuntimeError("invalid autograd function subclass")

        @staticmethod
        def apply() -> None:
            return None

    monkeypatch.setattr(torch_backend._STATE, "function", RejectingFunction)
    torch_backend.install_function_apply_wrappers()

    with pytest.raises(RuntimeError, match="invalid autograd function subclass"):

        class InvalidFunction(RejectingFunction):
            pass


def test_autograd_function_subclass_forwards_kwargs_and_wraps_future_apply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forwarded: List[str] = []

    class PlainFunction:
        @classmethod
        def __init_subclass__(cls, registration: str = "", **kwargs: object) -> None:
            super().__init_subclass__(**kwargs)
            forwarded.append(registration)

        @staticmethod
        def apply() -> None:
            return None

    monkeypatch.setattr(torch_backend._STATE, "function", PlainFunction)
    torch_backend.install_function_apply_wrappers()

    class ValidFunction(PlainFunction, registration="forwarded"):
        @staticmethod
        def apply() -> str:
            return "applied"

    assert forwarded == ["forwarded"]
    assert getattr(ValidFunction.apply, "_roctx_wrapped", False)


def test_autograd_function_read_only_subclass_hook_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReadOnlyMeta(type):
        def __setattr__(cls, name: str, value: object) -> None:
            if name == "__init_subclass__":
                raise TypeError("read-only class")
            super().__setattr__(name, value)

    class ReadOnlyFunction(metaclass=ReadOnlyMeta):
        @staticmethod
        def apply() -> None:
            return None

    warnings: List[str] = []
    monkeypatch.setattr(torch_backend._STATE, "function", ReadOnlyFunction)
    monkeypatch.setattr(
        torch_backend,
        "console_warning",
        lambda _category, message: warnings.append(message),
    )

    torch_backend.install_function_apply_wrappers()

    assert "Could not install torch.autograd.Function subclass hook" in warnings[0]


def test_extra_structural_wrappers_wrap_module_and_cuda_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pushes: List[str] = []
    pops: List[str] = []
    launcher_calls: List[str] = []

    def operation(value: int) -> int:
        return value + 1

    def backward() -> str:
        return "backward"

    def grad() -> str:
        return "grad"

    class FakeEvent:
        pass

    class FakeStream:
        def __init__(self) -> None:
            self.initialized = True

    fake_module = SimpleNamespace(operation=operation)
    fake_autograd = SimpleNamespace(backward=backward, grad=grad)
    fake_cuda = SimpleNamespace(Event=FakeEvent, Stream=FakeStream)
    monkeypatch.setattr(
        torch_backend,
        "EXTRA_STRUCTURAL_WRAPS",
        (
            ("fake.module", "operation", "torch.fake.operation"),
            ("torch.autograd", "backward", "torch.autograd.backward"),
            ("torch.autograd", "grad", "torch.autograd.grad"),
        ),
    )
    monkeypatch.setattr(torch_backend._STATE, "cuda_mod", fake_cuda)
    monkeypatch.setattr(
        torch_backend.importlib,
        "import_module",
        lambda module_path: (
            fake_autograd if module_path == "torch.autograd" else fake_module
        ),
    )
    monkeypatch.setattr(
        torch_backend.core,
        "resolve_user_caller_location",
        lambda: "test.py:1",
    )
    monkeypatch.setattr(
        torch_backend,
        "_push_scope",
        lambda marker, _context, **_kwargs: pushes.append(marker),
    )
    monkeypatch.setattr(torch_backend, "_pop_scope", lambda: pops.append("pop"))
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "push_launcher_tid",
        lambda: launcher_calls.append("push") or True,
    )
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "pop_launcher_tid",
        lambda: launcher_calls.append("pop") or True,
    )

    torch_backend.install_extra_structural_wrappers()

    assert fake_module.operation(1) == 2
    assert fake_autograd.backward() == "backward"
    assert fake_autograd.grad() == "grad"
    FakeEvent()
    stream = FakeStream()
    assert stream.initialized
    assert pushes == [
        "torch.fake.operation",
        "torch.autograd.backward",
        "torch.autograd.grad",
        "torch.cuda.Event",
        "torch.cuda.Stream",
    ]
    assert len(pops) == len(pushes)
    assert launcher_calls == ["push", "pop", "push", "pop"]


@pytest.mark.parametrize("native_collector_installed", [True, False])
def test_backend_install_selects_native_or_dispatcher_tier(
    monkeypatch: pytest.MonkeyPatch,
    native_collector_installed: bool,
) -> None:
    calls: List[str] = []
    fallback_calls: List[str] = []
    actual_extra_installer = torch_backend.install_extra_structural_wrappers
    installers = (
        "patch_distributed_collectives",
        "patch_process_group_methods",
        "patch_cuda_graph",
        "patch_compile_callable",
        "install_tensor_backward_wrapper",
        "inject_roctx_into_optimizer",
        "install_function_apply_wrappers",
        "install_tensor_method_wrappers",
        "install_extra_structural_wrappers",
        "inject_roctx_into_model",
        "inject_roctx_into_module_methods",
    )

    monkeypatch.setattr(torch_backend, "_ROCTX_AVAILABLE", True)
    monkeypatch.setattr(torch_backend, "_resolve_torch", lambda: True)
    monkeypatch.setattr(torch_backend, "EXTRA_STRUCTURAL_WRAPS", ())
    monkeypatch.setattr(torch_backend._STATE, "cuda_mod", None)
    monkeypatch.setattr(
        torch_backend.torch_trace_collector,
        "install",
        lambda: native_collector_installed,
    )
    monkeypatch.setattr(
        torch_backend,
        "_emit_python_tier_fallback_warning",
        lambda: fallback_calls.append("warning"),
    )
    monkeypatch.setattr(
        torch_backend,
        "install_dispatcher_hook",
        lambda: fallback_calls.append("dispatcher"),
    )
    monkeypatch.setattr(torch_backend, "console_log", lambda *_args: None)
    for installer in installers:
        if installer == "install_extra_structural_wrappers":
            continue
        monkeypatch.setattr(
            torch_backend,
            installer,
            lambda installer=installer: calls.append(installer),
        )

    def run_extra_structural_wrappers() -> None:
        calls.append("install_extra_structural_wrappers")
        actual_extra_installer()

    monkeypatch.setattr(
        torch_backend,
        "install_extra_structural_wrappers",
        run_extra_structural_wrappers,
    )

    torch_backend.TorchBackend().install()

    assert Counter(calls) == Counter(installers)
    expected_fallback_calls = (
        [] if native_collector_installed else ["warning", "dispatcher"]
    )
    assert fallback_calls == expected_fallback_calls
