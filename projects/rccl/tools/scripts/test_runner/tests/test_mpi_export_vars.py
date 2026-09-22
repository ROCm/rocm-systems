import unittest

from lib.test_executor import TestExecutor, collect_mpi_export_vars


class CollectMpiExportVarsTest(unittest.TestCase):
    def test_config_vars_are_exported_and_ambient_env_is_not(self):
        # A NCCL_/RCCL_ variable left in the launcher shell by an unrelated job
        # must not be forwarded to the ranks just because of its prefix.
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
        self.assertNotIn("NCCL_IB_QPS_PER_CONNECTION", got)
        self.assertNotIn("RCCL_IB_P2P_DISABLE_CTS", got)
        self.assertNotIn("LD_LIBRARY_PATH", got)
        self.assertNotIn("PATH", got)
        self.assertNotIn("HOME", got)

    def test_allowlisted_launcher_var_is_exported(self):
        # OpenMPI does not forward the launcher environment, so a value the
        # config takes from the operator's shell would otherwise apply to rank 0
        # alone and mismatch the remote ranks' QP count.
        merged = {"NCCL_NET": "IB-CAST"}
        env = {"NCCL_IB_QPS_PER_CONNECTION": "2", "NCCL_DEBUG_SUBSYS": "NET"}
        got = dict(collect_mpi_export_vars(env, merged, ["NCCL_IB_QPS_PER_CONNECTION"]))
        self.assertEqual(got["NCCL_IB_QPS_PER_CONNECTION"], "2")
        self.assertNotIn("NCCL_DEBUG_SUBSYS", got)

    def test_merged_env_wins_over_allowlisted_launcher_value(self):
        merged = {"NCCL_IB_QPS_PER_CONNECTION": "1"}
        env = {"NCCL_IB_QPS_PER_CONNECTION": "2"}
        pairs = list(collect_mpi_export_vars(env, merged, ["NCCL_IB_QPS_PER_CONNECTION"]))
        self.assertEqual(pairs, [("NCCL_IB_QPS_PER_CONNECTION", "1")])

    def test_allowlist_is_deduplicated_and_skips_unset_names(self):
        merged = {}
        env = {"NCCL_IB_HCA": "mlx5_0"}
        pairs = list(
            collect_mpi_export_vars(
                env, merged, ["NCCL_IB_HCA", "NCCL_IB_HCA", "NCCL_NOT_SET"]
            )
        )
        self.assertEqual(pairs, [("NCCL_IB_HCA", "mlx5_0")])

    def test_allowlist_cannot_reinstate_ld_library_path(self):
        # The caller exports LD_LIBRARY_PATH itself with build_dir first; the
        # launcher's bare value must never override that ordering.
        got = dict(
            collect_mpi_export_vars(
                {"LD_LIBRARY_PATH": "/stale/lib"}, {}, ["LD_LIBRARY_PATH"]
            )
        )
        self.assertEqual(got, {})

    def test_non_prefixed_allowlist_entry_is_honored(self):
        # The allowlist is explicit, so it is not limited to NCCL_/RCCL_ names.
        got = dict(collect_mpi_export_vars({"UCX_TLS": "rc"}, {}, ["UCX_TLS"]))
        self.assertEqual(got, {"UCX_TLS": "rc"})


class NormalizeExportEnvTest(unittest.TestCase):
    def test_defaults_to_empty_allowlist(self):
        self.assertEqual(TestExecutor._normalize_export_env(None, None, None), [])

    def test_unions_sources_in_order_without_duplicates(self):
        got = TestExecutor._normalize_export_env(
            ["NCCL_IB_HCA"], ["NCCL_IB_QPS_PER_CONNECTION", "NCCL_IB_HCA"], None
        )
        self.assertEqual(got, ["NCCL_IB_HCA", "NCCL_IB_QPS_PER_CONNECTION"])

    def test_accepts_comma_or_space_separated_string(self):
        got = TestExecutor._normalize_export_env("NCCL_IB_HCA, NCCL_IB_TC  NCCL_NET")
        self.assertEqual(got, ["NCCL_IB_HCA", "NCCL_IB_TC", "NCCL_NET"])

    def test_drops_blank_entries(self):
        self.assertEqual(TestExecutor._normalize_export_env(["", "  ", "NCCL_NET"]), ["NCCL_NET"])


if __name__ == "__main__":
    unittest.main()
