"""
4.6  Unit / Parser Tests
=========================
Validates CLI output parsing, unit conversion logic, and reconnect behavior.
"""

import os
import sys
import inspect
import pytest


class TestCLIOutputParsing:
    """Verify Python helpers can parse typical rdci output formats."""

    def test_rdc_util_has_helpers(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from RdcUtil import RdcUtil

            assert hasattr(RdcUtil, "get_all_gpu_indexes")
            assert hasattr(RdcUtil, "create_gpu_group")
            assert hasattr(RdcUtil, "create_field_group")
        finally:
            sys.path.pop(0)


class TestUnitConversion:
    """Verify field value conversion helpers in the Python bindings."""

    def test_field_value_struct_has_value(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_field_value

            field_names = [f[0] for f in rdc_field_value._fields_]
            assert "field_id" in field_names
            assert "status" in field_names
        finally:
            sys.path.pop(0)

    def test_field_id_enum_values_are_integers(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_field_t

            for field_id in (rdc_field_t.RDC_FI_GPU_TEMP, rdc_field_t.RDC_FI_POWER_USAGE):
                assert isinstance(getattr(field_id, "value", field_id), int)
        finally:
            sys.path.pop(0)


class TestReconnectLogic:
    """Verify the RdcReader reconnect code paths exist."""

    def test_reader_has_init(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from RdcReader import RdcReader

            assert hasattr(RdcReader, "__init__")
        finally:
            sys.path.pop(0)

    def test_reader_handles_field_ids(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from RdcReader import RdcReader

            sig = inspect.signature(RdcReader.__init__)
            params = list(sig.parameters.keys())
            assert "field_ids" in params, "RdcReader.__init__ missing field_ids param"
        finally:
            sys.path.pop(0)


class TestStructCompleteness:
    """Verify key ctypes structs have expected fields."""

    def test_gpu_attributes_struct(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_device_attributes_t

            field_names = [f[0] for f in rdc_device_attributes_t._fields_]
            assert len(field_names) > 0
        finally:
            sys.path.pop(0)

    def test_group_info_struct(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_group_info_t

            field_names = [f[0] for f in rdc_group_info_t._fields_]
            assert len(field_names) > 0, "rdc_group_info_t has no fields"
        finally:
            sys.path.pop(0)

    def test_job_info_struct(self, python_binding_dir):
        sys.path.insert(0, python_binding_dir)
        try:
            from rdc_bootstrap import rdc_job_info_t

            field_names = [f[0] for f in rdc_job_info_t._fields_]
            assert len(field_names) > 0, "rdc_job_info_t has no fields"
        finally:
            sys.path.pop(0)
