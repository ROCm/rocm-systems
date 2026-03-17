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

"""
Tests for region category filtering in rocpd summary generation.

This test suite verifies that the --region-categories flag properly filters
both view-based summaries and region-based summaries, preventing the bug where
all views were included regardless of specified categories.
"""

import pytest
import tempfile
import os
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

# Import the summary module
import sys

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent.parent / "source" / "lib" / "python")
)

from rocpd.summary import (
    create_summary_queries,
    create_summary_region_queries,
    generate_all_summaries,
    _view_matches_category,
)
from rocpd.importer import RocpdImportData


class TestViewMatchesCategory:
    """Test the helper function that matches view names to categories."""

    def test_exact_match(self):
        """Test exact match between view and category."""
        assert _view_matches_category("hip", "hip")
        assert _view_matches_category("kernel", "kernel")
        assert _view_matches_category("HIP", "hip")
        assert _view_matches_category("hip", "HIP")

    def test_plural_match(self):
        """Test that plural view names match singular categories."""
        assert _view_matches_category("kernels", "kernel")
        assert _view_matches_category("markers", "marker")
        assert _view_matches_category("KERNELS", "kernel")

    def test_prefix_match(self):
        """Test that view names with underscores match category prefixes."""
        assert _view_matches_category("kernel_trace", "kernel")
        assert _view_matches_category("hip_api", "hip")
        assert _view_matches_category("kernels_summary", "kernel")

    def test_plural_prefix_match(self):
        """Test plural with prefix."""
        assert _view_matches_category("kernels_by_name", "kernel")
        assert _view_matches_category("markers_filtered", "marker")

    def test_no_match(self):
        """Test cases that should not match."""
        assert not _view_matches_category("memory_copy", "kernel")
        assert not _view_matches_category("hip_api", "hsa")
        assert not _view_matches_category("scratch_memory", "marker")


class TestSummaryQueryFiltering:
    """Test that summary queries are correctly filtered by category."""

    @pytest.fixture
    def mock_connection(self):
        """Create a mock database connection with test views."""
        conn = Mock(spec=RocpdImportData)

        # Mock the execute method for temp view queries
        def mock_execute(query):
            result = Mock()
            if "sqlite_temp_master" in query:
                # Return mock view names
                result.fetchall.return_value = [
                    ("kernels",),
                    ("hip_api",),
                    ("memory_copy",),
                    ("scratch_memory",),
                    ("hsa_api",),
                ]
            return result

        # Mock cursor for column queries
        def mock_cursor():
            cursor_mock = Mock()
            # All views have name and duration columns
            cursor_mock.execute = Mock()
            cursor_mock.fetchall.return_value = [
                (0, "name", "TEXT", 0, None, 0),
                (1, "duration", "INTEGER", 0, None, 0),
                (2, "pid", "INTEGER", 0, None, 0),
                (3, "tid", "INTEGER", 0, None, 0),
            ]
            return cursor_mock

        conn.execute = mock_execute
        conn.cursor = mock_cursor
        return conn

    def test_no_filter_includes_all_views(self, mock_connection):
        """When only_view_categories is None, all eligible views should be included."""
        queries = create_summary_queries(
            mock_connection, by_rank=False, only_view_categories=None
        )

        # Should have summaries for all views (kernels, hip_api, memory_copy, scratch_memory, hsa_api)
        # Each gets one query (not by_rank)
        assert len(queries) == 5
        assert "kernels_summary" in queries
        assert "hip_api_summary" in queries
        assert "memory_copy_summary" in queries
        assert "scratch_memory_summary" in queries
        assert "hsa_api_summary" in queries

    def test_kernel_filter_only_includes_kernel_views(self, mock_connection):
        """When only_view_categories=['kernel'], only kernel-related views should be included."""
        queries = create_summary_queries(
            mock_connection, by_rank=False, only_view_categories=["kernel"]
        )

        # Should only have kernel summary
        assert len(queries) == 1
        assert "kernels_summary" in queries
        assert "hip_api_summary" not in queries
        assert "memory_copy_summary" not in queries

    def test_multiple_categories_filter(self, mock_connection):
        """When multiple categories specified, should include views matching any category."""
        queries = create_summary_queries(
            mock_connection, by_rank=False, only_view_categories=["kernel", "hip"]
        )

        # Should have kernel and hip summaries
        assert len(queries) == 2
        assert "kernels_summary" in queries
        assert "hip_api_summary" in queries
        assert "memory_copy_summary" not in queries
        assert "hsa_api_summary" not in queries

    def test_by_rank_with_filter(self, mock_connection):
        """Test that by_rank creates both regular and by-rank queries when filtering."""
        queries = create_summary_queries(
            mock_connection, by_rank=True, only_view_categories=["kernel"]
        )

        # Should have both regular and by-rank versions
        assert len(queries) == 2
        assert "kernels_summary" in queries
        assert "kernels_summary_by_rank" in queries

    def test_case_insensitive_filtering(self, mock_connection):
        """Test that category filtering is case-insensitive."""
        queries_upper = create_summary_queries(
            mock_connection, by_rank=False, only_view_categories=["KERNEL"]
        )

        queries_lower = create_summary_queries(
            mock_connection, by_rank=False, only_view_categories=["kernel"]
        )

        # Both should produce the same results
        assert set(queries_upper.keys()) == set(queries_lower.keys())


class TestGenerateAllSummariesIntegration:
    """Integration tests for the full summary generation workflow."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database with test data."""
        db_fd, db_path = tempfile.mkstemp(suffix=".db")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create test tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS processes (
                pid INTEGER PRIMARY KEY,
                hostname TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regions_and_samples (
                guid INTEGER,
                nid INTEGER,
                pid INTEGER,
                name TEXT,
                category TEXT,
                duration INTEGER
            )
        """
        )

        # Create temporary views
        cursor.execute(
            """
            CREATE TEMP VIEW kernels AS
            SELECT 'test_kernel' as name, 1000 as duration, 1 as pid, 1 as guid, 0 as nid
        """
        )

        cursor.execute(
            """
            CREATE TEMP VIEW hip_api AS
            SELECT 'hipMalloc' as name, 500 as duration, 1 as pid, 1 as guid, 0 as nid
        """
        )

        cursor.execute(
            """
            CREATE TEMP VIEW memory_copy AS
            SELECT 'memcpy_h2d' as name, 200 as duration, 1 as pid, 1 as guid, 0 as nid
        """
        )

        # Insert test process
        cursor.execute("INSERT INTO processes VALUES (1, 'localhost')")

        # Insert test region data
        cursor.execute(
            """
            INSERT INTO regions_and_samples VALUES
            (1, 0, 1, 'hip_region', 'rocm_hip', 100),
            (1, 0, 1, 'hsa_region', 'rocm_hsa', 150),
            (1, 0, 1, 'marker_test', 'MARKER_USER', 50)
        """
        )

        conn.commit()
        conn.close()

        yield db_path

        # Cleanup
        os.close(db_fd)
        os.unlink(db_path)

    def test_none_categories_includes_all_views_no_regions(self, temp_db):
        """Test that --region-categories NONE includes all views but no regions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            connection = RocpdImportData([temp_db])

            # Count queries by patching export_query
            exported_queries = {}

            def mock_export(
                conn, output_path, output_file, output_format, query_name, query
            ):
                exported_queries[query_name] = query

            with patch("rocpd.summary.export_query", side_effect=mock_export):
                generate_all_summaries(
                    connection,
                    region_categories=["NONE"],
                    output_path=tmpdir,
                    format="csv",
                )

            # Should have view summaries but no region summaries
            assert "kernels_summary" in exported_queries
            assert "hip_api_summary" in exported_queries
            assert "memory_copy_summary" in exported_queries

            # Should NOT have region summaries
            region_queries = [q for q in exported_queries.keys() if "rocm_" in q.lower()]
            assert len(region_queries) == 0

    def test_specific_category_filters_views_and_regions(self, temp_db):
        """Test that specific category filters both views and regions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            connection = RocpdImportData([temp_db])

            exported_queries = {}

            def mock_export(
                conn, output_path, output_file, output_format, query_name, query
            ):
                exported_queries[query_name] = query

            with patch("rocpd.summary.export_query", side_effect=mock_export):
                generate_all_summaries(
                    connection,
                    region_categories=["kernel"],
                    output_path=tmpdir,
                    format="csv",
                )

            # Should ONLY have kernel-related summaries
            assert "kernels_summary" in exported_queries

            # Should NOT have other view summaries
            assert "hip_api_summary" not in exported_queries
            assert "memory_copy_summary" not in exported_queries

            # Should NOT have region summaries (no kernel regions in test data)
            assert "rocm_hip_summary" not in exported_queries

    def test_multiple_categories_filter(self, temp_db):
        """Test that multiple categories include all matching views and regions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            connection = RocpdImportData([temp_db])

            exported_queries = {}

            def mock_export(
                conn, output_path, output_file, output_format, query_name, query
            ):
                exported_queries[query_name] = query

            with patch("rocpd.summary.export_query", side_effect=mock_export):
                generate_all_summaries(
                    connection,
                    region_categories=["kernel", "hip"],
                    output_path=tmpdir,
                    format="csv",
                )

            # Should have kernel and hip summaries
            assert "kernels_summary" in exported_queries
            assert "hip_api_summary" in exported_queries

            # Should NOT have memory_copy
            assert "memory_copy_summary" not in exported_queries

            # Should have hip region summary
            assert "rocm_hip_summary" in exported_queries


if __name__ == "__main__":
    pytest.main(["-v", __file__])
