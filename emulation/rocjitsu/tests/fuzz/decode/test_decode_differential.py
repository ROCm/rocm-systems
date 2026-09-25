#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "fuzz" / "decode"))
from decode_differential import (
    _git_revision,
    _inputs,
    canonicalize_text,
    compare_records,
    llvm_args,
    llvm_consumed_size,
    parse_llvm_output,
    run_one,
)


class LlvmArgsTest(unittest.TestCase):
    def test_selects_requested_target(self):
        self.assertIn("--mcpu=gfx950", llvm_args("gfx950"))
        self.assertIn("--mcpu=gfx1201", llvm_args("gfx1201"))
        self.assertIn("--mcpu=gfx1251", llvm_args("gfx1251"))

    def test_maps_architecture_to_representative_processor(self):
        self.assertIn("--mcpu=gfx908", llvm_args("cdna1"))
        self.assertIn("--mcpu=gfx1150", llvm_args("rdna3_5"))


class ParseLlvmOutputTest(unittest.TestCase):
    def test_parses_first_valid_instruction_and_size(self):
        output = (
            "\tv_add_f32_e32 v0, v1, v2 ; "
            "encoding: [0x01,0x05,0x00,0x02]\n"
            "\tv_illegal ; encoding: [0x00,0x00,0x00,0x00]\n"
        )
        self.assertEqual(
            parse_llvm_output(output, ""),
            {
                "status": "valid",
                "size": 4,
                "mnemonic": "v_add_f32_e32",
                "disassembly": "v_add_f32_e32 v0, v1, v2",
            },
        )

    def test_column_one_warning_rejects_first_instruction(self):
        warning = "<stdin>:1:1: warning: invalid instruction encoding\n0xff 0xff 0xff 0xff\n^\n"
        result = parse_llvm_output(
            "\ts_nop 0 ; encoding: [0x00,0x00,0x80,0xbf]\n", warning
        )
        self.assertEqual(result["status"], "invalid")

    def test_later_warning_does_not_reject_first_instruction(self):
        warning = "<stdin>:1:22: warning: invalid instruction encoding\ninput\n                     ^\n"
        result = parse_llvm_output(
            "\ts_nop 0 ; encoding: [0x00,0x00,0x80,0xbf]\n", warning
        )
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["size"], 4)

    def test_classifies_inline_operand_rejection(self):
        output = (
            "\tv_readlane_b32 s0, s0/*invalid register*/, s0 ; "
            "encoding: [0x00,0x00,0x00,0x00]\n"
        )
        result = parse_llvm_output(output, "")
        self.assertEqual(result["status"], "operand_rejected")
        self.assertIn("/*invalid register*/", result["annotated_disassembly"])
        self.assertEqual(
            compare_records(
                {
                    "status": "valid",
                    "size": 4,
                    "mnemonic": "v_readlane_b32",
                    "disassembly": "v_readlane_b32 s0, s0, s0",
                },
                result,
            ),
            ["llvm_operand_rejected"],
        )

    def test_operand_rejection_preserves_other_differences(self):
        llvm = {
            "status": "operand_rejected",
            "size": 8,
            "mnemonic": "v_movrels_b32_e64",
            "disassembly": "v_movrels_b32_e64 v0, s0",
            "annotated_disassembly": "v_movrels_b32_e64 v0, s0/*invalid register*/",
        }
        roc = {
            "status": "valid",
            "size": 4,
            "mnemonic": "v_movrels_b32",
            "disassembly": "v_movrels_b32 v0, s0",
        }
        self.assertEqual(
            compare_records(roc, llvm),
            [
                "llvm_operand_rejected",
                "size_mismatch",
                "mnemonic_mismatch",
                "text_mismatch",
            ],
        )


class LlvmConsumedSizeTest(unittest.TestCase):
    @mock.patch("decode_differential.subprocess.run")
    def test_uses_shortest_accepted_prefix_not_shown_encoding(self, run):
        invalid = subprocess.CompletedProcess(
            [], 0, "", "<stdin>:1:1: warning: invalid instruction encoding\n"
        )
        valid = subprocess.CompletedProcess(
            [],
            0,
            "\ts_add_nc_u64 s[4:5], s[0:1], 1 ; "
            "encoding: [0x00,0xfe,0x84,0xa9,0x01,0x00,0x00,0x00,"
            "0x00,0x00,0x00,0x00]\n",
            "",
        )
        run.side_effect = [invalid, valid]

        self.assertEqual(
            llvm_consumed_size(
                bytes(16),
                Path("llvm-mc"),
                1,
                "s_add_nc_u64 s[4:5], s[0:1], 1",
            ),
            8,
        )
        self.assertEqual(
            [len(call.kwargs["input"].split()) for call in run.call_args_list],
            [4, 8],
        )

    @mock.patch("decode_differential.subprocess.run")
    def test_rejects_prefix_that_decodes_as_a_different_instruction(self, run):
        invalid = subprocess.CompletedProcess(
            [], 0, "", "<stdin>:1:1: warning: invalid instruction encoding\n"
        )
        short = subprocess.CompletedProcess(
            [],
            0,
            "\tv_wmma_ld_scale_paired_b32 v0, v4 ; "
            "encoding: [0x00,0x00,0x35,0xcc,0x00,0x09,0x02,0x02]\n",
            "",
        )
        run.side_effect = [invalid, short]

        self.assertEqual(
            llvm_consumed_size(
                bytes(12),
                Path("llvm-mc"),
                1,
                "v_wmma_scale_f32_16x16x128_f8f6f4 v[6:13], v[18:33], "
                "v[52:67], 0, v0, v4",
            ),
            12,
        )

    @mock.patch("decode_differential.subprocess.run")
    def test_failed_prefix_makes_consumed_size_unknown(self, run):
        invalid = subprocess.CompletedProcess(
            [], 0, "", "<stdin>:1:1: warning: invalid instruction encoding\n"
        )
        crashed = subprocess.CompletedProcess([], -11, "", "stack trace")
        valid = subprocess.CompletedProcess(
            [],
            0,
            "\tv_fract_f32_dpp v0, v204 dpp8:[0,0,1,0,0,0,0,0] fi:1 ; "
            "encoding: [0xea,0x40,0x00,0x7e,0xcc,0x40,0x00,0x00]\n",
            "",
        )
        run.side_effect = [invalid, crashed, valid]
        failures = []

        self.assertEqual(
            llvm_consumed_size(
                bytes(16),
                Path("llvm-mc"),
                1,
                "v_fract_f32_dpp v0, v204 dpp8:[0,0,1,0,0,0,0,0] fi:1",
                failures,
            ),
            None,
        )
        self.assertEqual(
            failures, [{"size": 8, "returncode": -11, "stderr": "stack trace"}]
        )


class RunOneTest(unittest.TestCase):
    @mock.patch("decode_differential._run_llvm")
    @mock.patch("decode_differential.subprocess.run")
    def test_operand_rejection_preserves_prefix_tool_failure(self, run, run_llvm):
        data = bytes.fromhex("860061d7ff0c0302ff00000000000000")
        rocjitsu = {
            "status": "valid",
            "size": 12,
            "mnemonic": "v_readlane_b32",
            "disassembly": "v_readlane_b32 s0, s0, s0",
        }
        run.return_value = subprocess.CompletedProcess([], 0, json.dumps(rocjitsu), "")
        invalid = subprocess.CompletedProcess(
            [], 0, "", "<stdin>:1:1: warning: invalid instruction encoding\n"
        )
        crashed = subprocess.CompletedProcess([], -11, "", "stack trace")
        operand_rejected = subprocess.CompletedProcess(
            [],
            0,
            "\tv_readlane_b32 s0, s0/*invalid register*/, s0 ; "
            "encoding: [0x86,0x00,0x61,0xd7,0xff,0x0c,0x03,0x02,"
            "0xff,0x00,0x00,0x00]\n",
            "",
        )
        run_llvm.side_effect = [operand_rejected, invalid, crashed, operand_rejected]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prefix-crash.bin"
            path.write_bytes(data)
            result = run_one(
                path,
                Path("rj_decode_fuzz"),
                Path("llvm-mc"),
                "gfx1250",
                "manual-qualified",
            )

        self.assertEqual(
            result["categories"],
            ["llvm_operand_rejected", "llvm_prefix_tool_failure"],
        )
        self.assertEqual(result["llvm"]["shown_encoding_size"], 12)
        self.assertIsNone(result["llvm"]["size"])
        self.assertEqual(
            result["llvm"]["prefix_tool_failures"],
            [{"size": 8, "returncode": -11, "stderr": "stack trace"}],
        )
        self.assertEqual(
            [len(call.args[1]) for call in run_llvm.call_args_list], [16, 4, 8, 12]
        )


class ComparisonTest(unittest.TestCase):
    def test_integer_spelling_and_whitespace_are_normalized(self):
        self.assertEqual(
            canonicalize_text("V_ADD_U32 v[0 : 1], 0x10"),
            canonicalize_text("v_add_u32  v[0:1],16"),
        )

    def test_modifier_order_remains_significant(self):
        self.assertNotEqual(
            canonicalize_text("v_add_f32 v0, v1 clamp mul:2"),
            canonicalize_text("v_add_f32 v0, v1 mul:2 clamp"),
        )

    def test_reports_structured_mismatch_categories(self):
        roc = {
            "status": "valid",
            "size": 12,
            "mnemonic": "v_fmamk_f64_e32",
            "disassembly": "v_fmamk_f64_e32 v[2:3], v[4:5], 0xc1f0000000000000, v[2:3]",
        }
        llvm = {
            "status": "valid",
            "size": 12,
            "mnemonic": "v_fmamk_f64",
            "disassembly": "v_fmamk_f64 v[2:3], v[4:5], 0xc1f00000, v[2:3]",
        }
        self.assertEqual(
            compare_records(roc, llvm),
            ["mnemonic_mismatch", "numeric_token_mismatch", "text_mismatch"],
        )

    def test_reports_acceptance_without_text_noise(self):
        self.assertEqual(
            compare_records({"status": "invalid"}, {"status": "valid"}),
            ["acceptance_mismatch"],
        )

    def test_reports_each_failed_process(self):
        self.assertEqual(
            compare_records(
                {"status": "tool_failure"},
                {"status": "tool_failure"},
            ),
            ["rocjitsu_tool_failure", "llvm_tool_failure"],
        )


class CorpusTest(unittest.TestCase):
    def test_ignores_hidden_afl_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary)
            queue_input = corpus / "id:000000"
            queue_input.write_bytes(bytes(16))
            (corpus / "README.txt").write_text("AFL metadata", encoding="utf-8")
            state = corpus / ".state" / "deterministic_done"
            state.mkdir(parents=True)
            (state / queue_input.name).write_bytes(b"")
            self.assertEqual(_inputs(None, corpus), [queue_input])


class RevisionTest(unittest.TestCase):
    def test_ignores_untracked_outputs_but_records_tracked_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
            subprocess.run(
                ["git", "-C", str(repository), "config", "user.name", "Test"],
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "config",
                    "user.email",
                    "test@example.com",
                ],
                check=True,
            )
            source = repository / "tracked.txt"
            source.write_text("original\n", encoding="utf-8")
            subprocess.run(
                ["git", "-C", str(repository), "add", "tracked.txt"], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-c",
                    "commit.gpgSign=false",
                    "-C",
                    str(repository),
                    "commit",
                    "--quiet",
                    "-m",
                    "initial",
                ],
                check=True,
            )

            (repository / "build-decode").mkdir()
            self.assertNotIn("+dirty", _git_revision(repository))

            source.write_text("modified\n", encoding="utf-8")
            self.assertTrue(_git_revision(repository).endswith("+dirty"))


if __name__ == "__main__":
    unittest.main()
