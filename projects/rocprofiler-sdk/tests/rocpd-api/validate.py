#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""Tests for rocprofiler-sdk-rocpd API: schema retrieval via RocpdSchema."""

import sys

import pytest


def test_schema_tables():
    """Load schema via RocpdSchema and assert tables SQL is non-empty and contains expected content."""
    from rocpd.schema import RocpdSchema

    schema = RocpdSchema(uuid="", guid="")
    assert schema.tables, "RocpdSchema.tables must be non-empty"
    assert "CREATE TABLE" in schema.tables, "Schema tables must contain CREATE TABLE"
    assert "rocpd_metadata" in schema.tables, "Schema tables must define rocpd_metadata"
    table_count = schema.tables.count("CREATE TABLE")
    print(f"  Tables found: {table_count}")


def test_schema_views():
    """Load schema via RocpdSchema and assert views SQL contains expected content."""
    from rocpd.schema import RocpdSchema

    schema = RocpdSchema(uuid="", guid="")
    assert schema.views, "RocpdSchema.views must be non-empty"
    assert "CREATE VIEW" in schema.views, "Schema views must contain CREATE VIEW"
    assert "kernels" in schema.views, "Schema views must define kernels view"
    view_count = schema.views.count("CREATE VIEW")
    print(f"  Views found: {view_count}")


def test_query_supported_schema_versions():
    """Test query_supported_schema_versions function."""

    from rocpd.schema import query_supported_schema_versions

    schema_versions = query_supported_schema_versions()
    assert schema_versions, "query_supported_schema_versions must be non-empty"
    assert (
        len(schema_versions) > 0
    ), "query_supported_schema_versions must contain at least one version"
    print(f"\n\n  Supported schema versions: {schema_versions}")


def test_load_all_schema_versions():
    from rocpd.schema import RocpdSchema, query_supported_schema_versions
    import re

    def schema_version_from_metadata_sql(sql: str):
        m = re.search(
            r'\(\s*"schema_version"\s*,\s*"([^"]*)"\s*\)',
            sql,
            flags=re.DOTALL,
        )
        return m.group(1) if m else None

    schema_versions = query_supported_schema_versions()
    assert schema_versions, "query_supported_schema_versions must be non-empty"
    assert (
        len(schema_versions) > 0
    ), "query_supported_schema_versions must contain at least one version"
    print(f"\n\n  Supported schema versions: {schema_versions}")
    for version in schema_versions:
        print(f"  Getting schema version: {version}")
        schema = RocpdSchema(uuid="", guid="", version=version)
        assert schema.tables, "RocpdSchema.tables must be non-empty"
        assert "CREATE TABLE" in schema.tables, "Schema tables must contain CREATE TABLE"
        assert schema.views, "RocpdSchema.views must be non-empty"
        assert "CREATE VIEW" in schema.views, "Schema views must contain CREATE VIEW"
        assert schema.metadata, "RocpdSchema.metadata must be non-empty"
        version_from_metadata = schema_version_from_metadata_sql(schema.metadata)
        assert version_from_metadata == str(
            version
        ), f"Schema version from metadata must match schema version. Requested: {version}, got: {version_from_metadata}"


if __name__ == "__main__":
    exit_code = pytest.main(["-x", __file__] + sys.argv[1:])
    sys.exit(exit_code)
