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

import logging
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

logging.TRACE = logging.DEBUG - 5
logging.addLevelName(logging.TRACE, "TRACE")


def trace_logger(message, *args, **kwargs):
    logging.log(logging.TRACE, message, *args, **kwargs)


setattr(logging, "trace", trace_logger)

from db_connector import DatabaseConnector

"""
Tests for the DatabaseConnector class that tests almost methods with initialization,
CSV import, database removal, and error handling.
The tests use mocks instead of a real MongoDB server for speed and reliability.
"""


class TestDatabaseConnector:
    @pytest.fixture
    def mock_args_import(self):
        """Mock arguments for import operation"""
        args = Mock()
        args.username = "test_user"
        args.password = "test_pass"
        args.host = "localhost"
        args.port = 27017
        args.team = "test_team"
        args.workload = "/app/tests/workloads/device_filter/MI100"
        args.upload = True
        args.remove = False
        args.kernel_verbose = False
        return args

    @pytest.fixture
    def mock_args_remove(self):
        """Mock arguments for remove operation"""
        args = Mock()
        args.username = "test_user"
        args.password = "test_pass"
        args.host = "localhost"
        args.port = 27017
        args.team = "test_team"
        args.workload = "rocprofiler-compute_test_team_workload_mi100"
        args.upload = False
        args.remove = True
        args.kernel_verbose = False
        return args

    def test_init(self, mock_args_import):
        """Test DatabaseConnector initialization"""
        connector = DatabaseConnector(mock_args_import)

        assert connector.args == mock_args_import
        assert isinstance(connector.cache, dict)
        assert len(connector.cache) == 0

        expected_connection_info = {
            "username": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": "27017",
            "team": "test_team",
            "workload": "/app/tests/workloads/device_filter/MI100",
            "db": None,
        }
        assert connector.connection_info == expected_connection_info
        assert connector.interaction_type is None
        assert connector.client is None

    @patch("db_connector.pd.read_csv")
    @patch("db_connector.Path")
    def test_prep_import_success(self, mock_path, mock_read_csv, mock_args_import):
        """Test successful prep_import"""
        # Setup mocks
        mock_path.return_value.joinpath.return_value = "/fake/path/sysinfo.csv"
        mock_path.return_value.is_file.return_value = True

        mock_sysinfo = pd.DataFrame({
            "gpu_model": ["MI100 "],
            "workload_name": [" test_workload"],
        })
        mock_read_csv.return_value = mock_sysinfo

        connector = DatabaseConnector(mock_args_import)
        connector.prep_import()

        expected_db = "rocprofiler-compute_test_team_test_workload_MI100"
        assert connector.connection_info["db"] == expected_db

    @patch("db_connector.pd.read_csv")
    @patch("db_connector.Path")
    def test_prep_import_missing_file(self, mock_path, mock_read_csv, mock_args_import):
        """Test prep_import when sysinfo.csv is missing"""
        mock_path.return_value.joinpath.return_value = "/fake/path/sysinfo.csv"
        mock_path.return_value.is_file.return_value = False

        connector = DatabaseConnector(mock_args_import)

        with patch(
            "db_connector.console_error", side_effect=SystemExit(1)
        ) as mock_console_error:
            with pytest.raises(SystemExit):
                connector.prep_import()

            mock_console_error.assert_called_with(
                "database", "Unable to parse SoC and/or workload name from sysinfo.csv"
            )

    @patch("db_connector.pd.read_csv")
    @patch("db_connector.Path")
    def test_prep_import_key_error(self, mock_path, mock_read_csv, mock_args_import):
        """Test prep_import when required fields are missing"""
        mock_path.return_value.joinpath.return_value = "/fake/path/sysinfo.csv"
        mock_path.return_value.is_file.return_value = True

        mock_sysinfo = pd.DataFrame({"other_column": ["value"]})
        mock_read_csv.return_value = mock_sysinfo

        connector = DatabaseConnector(mock_args_import)

        with patch(
            "db_connector.console_error", side_effect=SystemExit(1)
        ) as mock_console_error:
            with pytest.raises(SystemExit):
                connector.prep_import()

            assert mock_console_error.called
            error_call = mock_console_error.call_args[0][0]
            assert "Outdated workload" in error_call

    @patch("db_connector.tqdm")
    @patch("db_connector.os.listdir")
    @patch("db_connector.console_log")
    @patch("db_connector.console_warning")
    @patch("db_connector.kernel_name_shortener")
    @patch("db_connector.MongoClient")
    @patch("db_connector.pd.read_csv")
    def test_db_import_success(
        self,
        mock_read_csv,
        mock_mongo_client,
        mock_kernel_shortener,
        mock_console_warning,
        mock_console_log,
        mock_listdir,
        mock_tqdm,
        mock_args_import,
    ):
        """Test successful database import"""
        mock_listdir.return_value = ["test_data.csv", "empty_file.csv", "non_csv.txt"]
        mock_tqdm.return_value = mock_listdir.return_value

        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.side_effect = [test_df, pd.errors.EmptyDataError()]

        mock_client_instance = MagicMock()
        mock_db = MagicMock()
        mock_collection = MagicMock()
        mock_workload_db = MagicMock()
        mock_workload_col = MagicMock()

        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.__getitem__.side_effect = lambda x: {
            "rocprofiler-compute_test_team_test_workload_MI100": mock_db,
            "workload_names": mock_workload_db,
        }.get(x, mock_db)
        mock_db.__getitem__.return_value = mock_collection
        mock_workload_db.__getitem__.return_value = mock_workload_col

        connector = DatabaseConnector(mock_args_import)
        connector.connection_info["workload"] = "/fake/workload/path"
        connector.client = mock_client_instance

        with patch.object(connector, "prep_import") as mock_prep:
            mock_prep.return_value = None
            connector.connection_info["db"] = (
                "rocprofiler-compute_test_team_test_workload_MI100"
            )

            connector.db_import()

        mock_collection.insert_many.assert_called_once()
        mock_workload_col.replace_one.assert_called_once()

    @patch("db_connector.console_log")
    def test_db_remove_success(self, mock_console_log, mock_args_remove):
        """Test successful database removal"""
        mock_client = MagicMock()
        mock_db_to_remove = MagicMock()
        mock_workload_names_db = MagicMock()
        mock_names_col = MagicMock()

        mock_client.__getitem__.side_effect = lambda x: {
            "rocprofiler-compute_test_team_workload_mi100": mock_db_to_remove,
            "workload_names": mock_workload_names_db,
        }[x]
        mock_workload_names_db.__getitem__.return_value = mock_names_col
        mock_db_to_remove.list_collection_names.return_value = ["col1", "col2"]

        connector = DatabaseConnector(mock_args_remove)
        connector.client = mock_client

        connector.db_remove()

        mock_client.drop_database.assert_called_once_with(mock_db_to_remove)
        mock_names_col.delete_many.assert_called_once_with({
            "name": "rocprofiler-compute_test_team_workload_mi100"
        })

    def test_pre_processing_no_action_specified(self, mock_args_import):
        """Test pre_processing when neither upload nor remove is specified"""
        mock_args_import.upload = False
        mock_args_import.remove = False

        connector = DatabaseConnector(mock_args_import)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    def test_pre_processing_remove_invalid_workload_name(self, mock_args_remove):
        """Test pre_processing remove with invalid workload name"""
        mock_args_remove.workload = "invalid_name"

        connector = DatabaseConnector(mock_args_remove)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    def test_pre_processing_remove_missing_host_username(self, mock_args_remove):
        """Test pre_processing remove with missing host/username"""
        mock_args_remove.host = None
        mock_args_remove.username = None

        connector = DatabaseConnector(mock_args_remove)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    def test_pre_processing_remove_protected_database(self, mock_args_remove):
        """Test pre_processing remove with protected database names"""
        mock_args_remove.workload = "admin"

        connector = DatabaseConnector(mock_args_remove)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    @patch("db_connector.Path")
    @patch("db_connector.is_workload_empty")
    @patch("db_connector.getpass.getpass")
    @patch("db_connector.console_log")
    @patch("db_connector.MongoClient")
    def test_pre_processing_import_password_prompt_success(
        self,
        mock_mongo_client,
        mock_console_log,
        mock_getpass,
        mock_is_workload_empty,
        mock_path,
        mock_args_import,
    ):
        """Test pre_processing import with password prompt success"""
        mock_args_import.password = ""
        mock_getpass.return_value = "prompted_password"

        mock_path.return_value.absolute.return_value.is_dir.return_value = True
        mock_path.return_value.absolute.return_value.resolve.return_value = (
            "/resolved/path"
        )

        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.server_info.return_value = {}

        connector = DatabaseConnector(mock_args_import)
        connector.pre_processing()

        mock_getpass.assert_called_once()
        mock_console_log.assert_called_with("database", "Password received")

    @patch("db_connector.Path")
    @patch("db_connector.is_workload_empty")
    @patch("db_connector.MongoClient")
    def test_pre_processing_import_connection_failure(
        self, mock_mongo_client, mock_is_workload_empty, mock_path, mock_args_import
    ):
        """Test pre_processing import with MongoDB connection failure"""
        mock_path.return_value.absolute.return_value.is_dir.return_value = True
        mock_path.return_value.absolute.return_value.resolve.return_value = (
            "/resolved/path"
        )

        mock_client_instance = MagicMock()
        mock_mongo_client.return_value = mock_client_instance
        mock_client_instance.server_info.side_effect = Exception("Connection failed")

        connector = DatabaseConnector(mock_args_import)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    @patch("db_connector.Path")
    @patch("db_connector.is_workload_empty")
    def test_pre_processing_import_missing_required_fields(
        self, mock_is_workload_empty, mock_path, mock_args_import
    ):
        """Test pre_processing import with missing required fields"""
        mock_args_import.host = None

        connector = DatabaseConnector(mock_args_import)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    @patch("db_connector.Path")
    def test_pre_processing_import_invalid_workload_path(
        self, mock_path, mock_args_import
    ):
        """Test pre_processing import with invalid workload path"""
        mock_path.return_value.absolute.return_value.is_dir.return_value = False

        connector = DatabaseConnector(mock_args_import)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()

    def test_pre_processing_import_team_name_too_long(self, mock_args_import):
        """Test pre_processing import with team name exceeding limit"""
        mock_args_import.team = "this_team_name_is_way_too_long"

        connector = DatabaseConnector(mock_args_import)

        with patch("db_connector.console_error", side_effect=SystemExit(1)):
            with pytest.raises(SystemExit):
                connector.pre_processing()


class TestDatabaseConnectorIntegration:
    """Simple integration test"""

    @patch("db_connector.Path")
    @patch("db_connector.pd.read_csv")
    def test_prep_import_with_real_workload_path(self, mock_read_csv, mock_path):
        """Test prep_import with actual workload path structure"""
        args = Mock()
        args.username = "test_user"
        args.password = "test_pass"
        args.host = "localhost"
        args.port = 27017
        args.team = "test_team"
        args.workload = "/app/tests/workloads/device_filter/MI100"
        args.upload = True
        args.remove = False
        args.kernel_verbose = False

        mock_path.return_value.joinpath.return_value = (
            "/app/tests/workloads/device_filter/MI100/sysinfo.csv"
        )
        mock_path.return_value.is_file.return_value = True

        mock_sysinfo = pd.DataFrame({
            "gpu_model": ["MI100"],
            "workload_name": ["device_filter"],
        })
        mock_read_csv.return_value = mock_sysinfo

        connector = DatabaseConnector(args)
        connector.prep_import()

        expected_db = "rocprofiler-compute_test_team_device_filter_MI100"
        assert connector.connection_info["db"] == expected_db


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
