import os
import unittest
from unittest import mock

from lib.test_config import expand_env_vars


class ConfigExpansionTest(unittest.TestCase):
    def test_colon_dash_uses_default_for_empty_variable(self):
        with mock.patch.dict("os.environ", {"RCCL_TESTS_DIR": ""}, clear=False):
            expanded = expand_env_vars(
                "${RCCL_TESTS_DIR:-/workspace/rccl-tests}/build"
            )

        self.assertEqual(expanded, "/workspace/rccl-tests/build")

    def test_colon_dash_prefers_set_variable(self):
        with mock.patch.dict("os.environ", {"RCCL_TESTS_DIR": "/real/path"}, clear=False):
            expanded = expand_env_vars(
                "${RCCL_TESTS_DIR:-/workspace/rccl-tests}/build"
            )

        self.assertEqual(expanded, "/real/path/build")

    def test_colon_dash_uses_default_when_variable_unset(self):
        env = {k: v for k, v in os.environ.items() if k != "RCCL_TESTS_DIR"}
        with mock.patch.dict("os.environ", env, clear=True):
            expanded = expand_env_vars(
                "${RCCL_TESTS_DIR:-/workspace/rccl-tests}/build"
            )

        self.assertEqual(expanded, "/workspace/rccl-tests/build")


if __name__ == "__main__":
    unittest.main()
