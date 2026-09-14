import unittest

from lib.test_executor import collect_mpi_export_vars


class CollectMpiExportVarsTest(unittest.TestCase):
    def test_json_vars_win_and_leftover_nccl_is_exported(self):
        merged = {"NCCL_NET": "IB", "NCCL_IB_TC": "104", "LD_LIBRARY_PATH": "/opt/rocm/lib"}
        env = {
            "PATH": "/usr/bin",
            "NCCL_NET": "IB",
            "NCCL_IB_TC": "104",
            "NCCL_IB_QPS_PER_CONNECTION": "2",
            "RCCL_IB_P2P_DISABLE_CTS": "1",
            "HOME": "/home/user",
        }
        got = dict(collect_mpi_export_vars(env, merged))
        self.assertEqual(got["NCCL_NET"], "IB")
        self.assertEqual(got["NCCL_IB_TC"], "104")
        self.assertEqual(got["NCCL_IB_QPS_PER_CONNECTION"], "2")
        self.assertEqual(got["RCCL_IB_P2P_DISABLE_CTS"], "1")
        self.assertNotIn("LD_LIBRARY_PATH", got)
        self.assertNotIn("PATH", got)
        self.assertNotIn("HOME", got)

    def test_merged_env_overrides_parent_leftover(self):
        merged = {"NCCL_IB_QPS_PER_CONNECTION": "1"}
        env = {"NCCL_IB_QPS_PER_CONNECTION": "2"}
        got = dict(collect_mpi_export_vars(env, merged))
        self.assertEqual(got["NCCL_IB_QPS_PER_CONNECTION"], "1")


if __name__ == "__main__":
    unittest.main()
