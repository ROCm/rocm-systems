#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import consan_validation as validation
import consan_validation_faults as validation_faults


class ConSanValidationTargetAdmissionTest(unittest.TestCase):
    def test_gfx1100_rocblas_fault_uses_the_native_singleton_publication(self) -> None:
        catalog = Path(__file__).with_name('consan_validation_faults_gfx1100.json')
        workload = validation._workload_for_target('gfx1100', 'rocblas-sgemm-square-64')
        cases = {
            'default': ('default', {'RJ_CONSAN_PRESET': 'default'}),
            'high': ('default', {'RJ_CONSAN_PRESET': 'high'}),
            'wg1-cell256': (
                'default',
                {
                    'RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE': '1',
                    'RJ_CONSAN_CELL_SAMPLE_STRIDE': '256',
                },
            ),
            'sc-sleep15': (
                'supercollider',
                {'RJ_CONSAN_SC_DELAY_MODE': 'sleep', 'RJ_CONSAN_SC_DELAY': '15'},
            ),
        }
        for suffix, delay, reads_only in (
            ('sc-sleep15-reads', '15', '1'),
            ('sc-sleep127-all', '127', '0'),
            ('sc-sleep127-reads', '127', '1'),
        ):
            cases[suffix] = (
                'supercollider',
                {
                    'RJ_CONSAN_SC_DELAY_MODE': 'sleep',
                    'RJ_CONSAN_SC_DELAY': delay,
                    'RJ_CONSAN_SC_DELAY_READS_ONLY': reads_only,
                },
            )
        for suffix, (profile, controls) in cases.items():
            with self.subTest(suffix=suffix):
                fault = validation_faults._load_fault(
                    catalog, 'gfx1100', workload, f'drop-initial-publication-{suffix}'
                )
                environment = fault['environment']
                self.assertEqual(environment['RJ_CONSAN_FAULT_DROP_BARRIER'], '1')
                identity = environment['RJ_CONSAN_FAULT_SITE_IDENTITY']
                self.assertIn('fnv1a64:3b148d41a0df9e19|', identity)
                self.assertIn('|pc=0x00000000002002d8|mnemonic=s_barrier|', identity)
                self.assertNotIn(
                    'RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY', environment
                )
                policy, trials = validation_faults._fault_trials(fault, profile)
                self.assertEqual(policy['detector'], 'statistical')
                self.assertEqual(policy['minimum_detections'], 6)
                self.assertEqual(trials, [controls] * 8)
                self.assertEqual(
                    fault['reach_witness']['kind'], 'reviewed-unconditional-final-isa'
                )

    def test_gfx1100_pytorch_faults_preserve_reviewed_sites_and_trials(self) -> None:
        catalog = Path(__file__).with_name('consan_validation_faults_gfx1100.json')
        cases = [
            (
                'pytorch-torch-mode',
                'mode-initial-lds-publication-high',
                'bb0ede9b5d4128a0',
                '2e56c',
                0,
                'high',
            ),
            (
                'pytorch-torch-sort',
                'sort-first-lds-publication-high',
                '2ab4471de865fd8f',
                '29a9f4',
                0,
                'high',
            ),
            (
                'pytorch-norm-softmax',
                'softmax-lds-publication-high',
                'dbee9568f8d7b8d5',
                '1471bc',
                2,
                'high',
            ),
        ]
        cases.extend(
            (
                'pytorch-norm-softmax',
                f'norm-lds-publication-{preset}',
                '2a88a7a8ddc0e00b',
                '309ca0',
                0,
                preset,
            )
            for preset in ('high', 'higher', 'max')
        )
        for workload_id, fault_id, fingerprint, pc, occurrence, preset in cases:
            with self.subTest(fault=fault_id):
                workload = validation._workload_for_target('gfx1100', workload_id)
                fault = validation_faults._load_fault(
                    catalog, 'gfx1100', workload, fault_id
                )
                environment = fault['environment']
                self.assertEqual(environment['RJ_CONSAN_FAULT_DROP_BARRIER'], '1')
                identity = environment['RJ_CONSAN_FAULT_SITE_IDENTITY']
                self.assertTrue(identity.startswith(f'fnv1a64:{fingerprint}|kernel='))
                self.assertTrue(
                    identity.endswith(
                        f'|kind=barrier|pc=0x{int(pc, 16):016x}|mnemonic=s_barrier|'
                        f'occurrence={occurrence}'
                    )
                )
                if workload_id == 'pytorch-torch-mode':
                    self.assertEqual(
                        environment['RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY'],
                        identity.replace('|kind=barrier|', '|event=barrier|')
                        + '|sequence=singleton',
                    )
                    companion = identity.replace('0002e56c', '0002e578').replace(
                        'occurrence=0', 'occurrence=1'
                    )
                    self.assertEqual(
                        environment['RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY'],
                        companion,
                    )
                    self.assertEqual(
                        environment[
                            'RJ_CONSAN_FAULT_BARRIER_COMPANION_SEQUENCE_IDENTITY'
                        ],
                        companion.replace('|kind=barrier|', '|event=barrier|')
                        + '|sequence=singleton',
                    )
                else:
                    self.assertEqual(
                        set(environment),
                        {
                            'RJ_CONSAN_FAULT_DROP_BARRIER',
                            'RJ_CONSAN_FAULT_SITE_IDENTITY',
                        },
                    )
                # Reach is established by runtime diagnostics, not assumed for misses.
                self.assertNotIn('reach_witness', fault)
                self.assertEqual(set(fault['profiles']), {'default'})
                policy, trials = validation_faults._fault_trials(fault, 'default')
                self.assertEqual(policy['detector'], 'statistical')
                self.assertEqual(policy['minimum_detections'], 6)
                self.assertEqual(policy['oracle'], 'any')
                self.assertEqual(trials, [{'RJ_CONSAN_PRESET': preset}] * 8)

    def test_gfx1100_explain_includes_all_p0_fault_campaigns(self) -> None:
        expected_counts = {
            'rocblas-sgemm-square-64': 7,
            'pytorch-torch-mode': 1,
            'pytorch-torch-sort': 1,
            'pytorch-norm-softmax': 4,
        }
        audit = validation._explain_contract(
            Path('/workspace'),
            'gfx1100',
            tuple(expected_counts),
            ('default', 'supercollider'),
            Path(__file__).with_name('consan_validation_faults_gfx1100.json'),
        )
        for workload in audit['workloads']:
            with self.subTest(workload=workload['id']):
                self.assertEqual(workload['fault_spec_status'], 'reviewed-spec')
                self.assertEqual(
                    len(workload['faults']), expected_counts[workload['id']]
                )
                for fault in workload['faults']:
                    profile = (
                        'supercollider' if '-sc-sleep' in fault['id'] else 'default'
                    )
                    expectation = next(
                        entry
                        for entry in fault['profile_expectations']
                        if entry['profile'] == profile
                    )
                    self.assertEqual(expectation['trial_count'], 8)
                    self.assertEqual(expectation['minimum_detections'], 6)
                    if fault['id'] == 'drop-initial-publication-wg1-cell256':
                        for trial in expectation['trials']:
                            settings = {
                                entry['name']: entry
                                for entry in trial['effective_settings']
                            }
                            for name, value in (
                                ('RJ_CONSAN_WORKGROUP_SAMPLE_STRIDE', '1'),
                                ('RJ_CONSAN_CELL_SAMPLE_STRIDE', '256'),
                            ):
                                self.assertEqual(settings[name]['value'], value)
                                self.assertEqual(
                                    settings[name]['category'], 'workload-tuning'
                                )
                                self.assertTrue(settings[name]['usability_exception'])

    def test_gfx1100_admits_initial_torch_workloads(self) -> None:
        workspace = Path('/workspace')
        python = workspace / 'consan-pytorch-venv/bin/python'
        for workload_id in (
            'pytorch-torch-mode',
            'pytorch-torch-sort',
            'pytorch-norm-softmax',
        ):
            with self.subTest(workload=workload_id):
                workload = validation._workload_for_target('gfx1100', workload_id)
                self.assertEqual(workload.kind, 'pytorch')
                with mock.patch.dict(
                    'os.environ', {'CONSAN_VALIDATION_PYTORCH_PYTHON': str(python)}
                ):
                    command = validation._workload_command(
                        workspace,
                        'gfx1100',
                        workload,
                        'clean',
                        workspace / 'result.json',
                    )
                self.assertEqual(command[0], str(python))
                self.assertIn(workload_id.removeprefix("pytorch-"), command)
                self.assertIn('--repetitions', command)
                self.assertEqual(command[command.index('--repetitions') + 1], '1')

    def test_rocblas_sgemm_uses_the_selected_native_target(self) -> None:
        workspace = Path("/workspace")
        workload_id = "rocblas-sgemm-square-64"
        for target in ("gfx1100", "gfx950"):
            with self.subTest(target=target):
                workload = validation._workload_for_target(target, workload_id)
                executable = workspace / (
                    f"rocjitsu-test-corpus-build/kernels-{target}-rocblas/cases/"
                    "rocblas/rocblas_sgemm"
                )
                manifest = {
                    row["id"]: row for row in validation._manifest(target)["workloads"]
                }
                self.assertEqual(
                    workspace / manifest[workload_id]["relative_path"], executable
                )
                self.assertEqual(
                    validation._input_files(workspace, target, workload)["executable"],
                    executable,
                )
                for phase in ("clean", "overhead", "fault"):
                    with self.subTest(phase=phase):
                        self.assertEqual(
                            validation._workload_command(
                                workspace,
                                target,
                                workload,
                                phase,
                                workspace / "result.json",
                            ),
                            [
                                str(executable),
                                "--gtest_filter=RocblasGemmTest.Square_64x64",
                            ],
                        )

    def test_gfx1100_admits_registered_native_gtests(self) -> None:
        expected = {
            "d128-block": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_d128_attention_block_test",
                "HipMoiGfx1100D128AttentionBlock.*",
            ),
            "d128-pressure": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_d128_attention_pressure_test",
                "HipMoiGfx1100D128AttentionPressure.*",
            ),
            "wmma-attention": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_wmma_attention_block_test",
                "HipMoiGfx1100WmmaAttentionBlock.*",
            ),
            "streamk-arrival": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_wmma_streamk_arrival_counter_test",
                "HipMoiGfx1100WmmaStreamKArrivalCounter."
                "ConSanOracleAcqRelFetchAddOrdersWmmaPartials",
            ),
            "tree-atomic-or": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_instrumented_gfx1100_wmma_streamk_tree_atomic_or_test",
                "HipMoiGfx1100WmmaStreamKTreeAtomicOr."
                "ConSanOracleAcqRelBitmaskOrdersWmmaPartials",
            ),
            "jakub-attention": (
                "hip-moi-build-gfx1100-tests/tests/"
                "hip_moi_reference_gfx1100_jakub_matmul",
                "SafeFp16Packed/JakubGfx1100MatmulReference." "MatchesHostReference/*",
            ),
        }
        manifest_ids = {
            workload["id"] for workload in validation._manifest("gfx1100")["workloads"]
        }
        self.assertTrue(set(expected).issubset(manifest_ids))

        self.assertEqual(set(expected), set(validation.NATIVE_GTEST_WORKLOAD_IDS))
        for workload_id, (relative_path, clean_filter) in expected.items():
            with self.subTest(workload=workload_id):
                workload = validation._workload_for_target("gfx1100", workload_id)
                resolved = validation._resolved_workload("gfx1100", workload)
                self.assertEqual(resolved.relative_path, relative_path)
                self.assertEqual(resolved.clean_filter, clean_filter)
        for workload_id in ("streamk-arrival", "tree-atomic-or"):
            with self.subTest(workload=workload_id):
                workload = validation._workload_for_target("gfx1100", workload_id)
                self.assertEqual(
                    validation._fault_families("gfx1100", workload),
                    ("atomic-weaken-order",),
                )

    def test_gfx1100_barrier_faults_target_reviewed_publications(self) -> None:
        catalog = Path(__file__).with_name("consan_validation_faults_gfx1100.json")
        expected = {
            "d128-block": (
                "barrier-drop-k-publication-group",
                "fnv1a64:43b0a96f4ab07982",
                "pc=0x0000000000054844",
                "pc=0x0000000000054850",
            ),
            "d128-pressure": (
                "barrier-drop-kv-publication-group",
                "fnv1a64:e969c5a2e1a058f7",
                "pc=0x000000000003ac0c",
                "pc=0x000000000003ac18",
            ),
            "wmma-attention": (
                "barrier-drop-kv-publication-group",
                "fnv1a64:f8a2c2abd1d454c2",
                "pc=0x0000000000015530",
                "pc=0x000000000001553c",
            ),
            "jakub-attention": (
                "barrier-drop-load-compute-publication",
                "fnv1a64:60f9f11b7f717a09",
                "pc=0x000000000000068c",
                None,
            ),
        }

        for workload_id, (
            fault_prefix,
            code_object,
            primary_pc,
            companion_pc,
        ) in expected.items():
            grouped = companion_pc is not None
            workload = validation.WORKLOAD_BY_ID[workload_id]
            with self.subTest(workload=workload_id, profile="default"):
                for preset in ("default", "high", "higher", "max"):
                    fault = validation_faults._load_fault(
                        catalog,
                        "gfx1100",
                        workload,
                        f"{fault_prefix}-preset-{preset}",
                    )
                    environment = fault["environment"]
                    self.assertEqual(environment["RJ_CONSAN_FAULT_DROP_BARRIER"], "1")
                    self.assertIn(
                        code_object, environment["RJ_CONSAN_FAULT_SITE_IDENTITY"]
                    )
                    self.assertIn(
                        primary_pc, environment["RJ_CONSAN_FAULT_SITE_IDENTITY"]
                    )
                    self.assertEqual(
                        "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY" in environment,
                        grouped,
                    )
                    self.assertEqual(
                        "RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY"
                        in environment,
                        grouped,
                    )
                    self.assertEqual(
                        "RJ_CONSAN_FAULT_BARRIER_COMPANION_SEQUENCE_IDENTITY"
                        in environment,
                        grouped,
                    )
                    if companion_pc is not None:
                        self.assertIn(
                            companion_pc,
                            environment[
                                "RJ_CONSAN_FAULT_BARRIER_COMPANION_SITE_IDENTITY"
                            ],
                        )
                    policy, trials = validation_faults._fault_trials(fault, "default")
                    self.assertEqual(policy["detector"], "statistical")
                    self.assertEqual(policy["minimum_detections"], 6)
                    self.assertEqual(policy["oracle"], "any")
                    self.assertEqual(trials, [{"RJ_CONSAN_PRESET": preset}] * 8)

            with self.subTest(workload=workload_id, profile="supercollider"):
                fault = validation_faults._load_fault(
                    catalog,
                    "gfx1100",
                    workload,
                    f"{fault_prefix}-sc-sleep-15",
                )
                policy, trials = validation_faults._fault_trials(fault, "supercollider")
                self.assertEqual(policy["detector"], "statistical")
                self.assertEqual(policy["minimum_detections"], 6)
                self.assertEqual(policy["oracle"], "any")
                self.assertEqual(
                    trials,
                    [
                        {
                            "RJ_CONSAN_SC_DELAY_MODE": "sleep",
                            "RJ_CONSAN_SC_DELAY": "15",
                        }
                    ]
                    * 8,
                )

    def test_gfx1100_fails_closed_without_native_registry(self) -> None:
        with (
            mock.patch.dict(validation.NATIVE_GTEST_TARGETS, {}, clear=True),
            mock.patch.dict(validation.NATIVE_GTEST_WORKLOAD_OVERRIDES, {}, clear=True),
        ):
            manifest_ids = {
                workload["id"]
                for workload in validation._manifest("gfx1100")["workloads"]
            }
            self.assertTrue(
                manifest_ids.isdisjoint(validation.NATIVE_GTEST_WORKLOAD_IDS)
            )

            for workload_id in validation.NATIVE_GTEST_WORKLOAD_IDS:
                with self.subTest(workload=workload_id):
                    with self.assertRaisesRegex(
                        validation.ValidationError,
                        f"gfx1100 manifest excludes workload: {workload_id}",
                    ):
                        validation._workload_for_target("gfx1100", workload_id)

            with self.assertRaisesRegex(
                validation.ValidationError,
                "gfx1100 gtest workload has no target-specific registry",
            ):
                validation._resolved_workload(
                    "gfx1100", validation.WORKLOAD_BY_ID["d128-block"]
                )


if __name__ == "__main__":
    unittest.main()
