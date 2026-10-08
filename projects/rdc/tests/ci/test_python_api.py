"""
4.2  Python API / RdcReader Tests
==================================
Validates the Python ctypes bindings, RdcReader, and RdcUtil modules.
"""

import importlib
import os
import sys
import pytest


# ---------------------------------------------------------------------------
# Module import / syntax checks (no GPU needed)
# ---------------------------------------------------------------------------
class TestPythonBindingImport:
    """Verify Python bindings can be imported without errors."""

    def test_rdc_bootstrap_importable(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            mod = importlib.import_module("rdc_bootstrap")
            assert hasattr(mod, "rdc_field_t")
            assert hasattr(mod, "rdc_status_t")
            assert hasattr(mod, "rdc_operation_mode_t")
        finally:
            sys.path.pop(0)

    def test_rdc_reader_importable(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            mod = importlib.import_module("RdcReader")
            assert hasattr(mod, "RdcReader")
        finally:
            sys.path.pop(0)

    def test_rdc_util_importable(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            mod = importlib.import_module("RdcUtil")
            assert hasattr(mod, "RdcUtil")
        finally:
            sys.path.pop(0)

    def test_rdc_prometheus_importable(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            importlib.import_module("rdc_prometheus")
        except ImportError as e:
            if "prometheus_client" in str(e):
                pytest.skip("prometheus_client not installed")
            raise
        finally:
            sys.path.pop(0)


# ---------------------------------------------------------------------------
# Enum / constant correctness (no GPU needed)
# ---------------------------------------------------------------------------
class TestBootstrapEnums:
    """Verify key enums and constants in rdc_bootstrap are correct."""

    def test_field_enum_has_gpu_temp(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_field_t

            assert hasattr(rdc_field_t, "RDC_FI_GPU_TEMP")
            assert hasattr(rdc_field_t, "RDC_FI_POWER_USAGE")
            assert hasattr(rdc_field_t, "RDC_FI_GPU_CLOCK")
            assert hasattr(rdc_field_t, "RDC_FI_GPU_UTIL")
            assert hasattr(rdc_field_t, "RDC_FI_GPU_MEMORY_USAGE")
        finally:
            sys.path.pop(0)

    def test_status_enum_has_success(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_status_t

            assert hasattr(rdc_status_t, "RDC_ST_OK")
        finally:
            sys.path.pop(0)

    def test_operation_mode_enum(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_operation_mode_t

            assert hasattr(rdc_operation_mode_t, "RDC_OPERATION_MODE_AUTO")
            assert hasattr(rdc_operation_mode_t, "RDC_OPERATION_MODE_MANUAL")
        finally:
            sys.path.pop(0)

    def test_metric_type_enum(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_metric_type_t

            assert hasattr(rdc_metric_type_t, "GAUGE")
            assert hasattr(rdc_metric_type_t, "COUNTER")
            assert hasattr(rdc_metric_type_t, "LABEL")
        finally:
            sys.path.pop(0)


# ---------------------------------------------------------------------------
# Python syntax / lint checks (no GPU needed)
# ---------------------------------------------------------------------------
class TestPythonQuality:
    """Basic quality checks on Python binding scripts."""

    def test_python_files_compile(self, python_binding_dir):
        errors = []
        for fname in os.listdir(python_binding_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(python_binding_dir, fname)
            try:
                with open(fpath) as f:
                    compile(f.read(), fpath, "exec")
            except SyntaxError as e:
                errors.append(f"{fname}: {e}")
        assert not errors, "Syntax errors found:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# RdcReader public surface
# ---------------------------------------------------------------------------
class TestPythonBindingSurface:
    """Verify the public RdcReader constructor is available."""

    def test_rdc_reader_is_callable(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from RdcReader import RdcReader

            assert callable(RdcReader), "RdcReader should be callable"
        finally:
            sys.path.pop(0)
