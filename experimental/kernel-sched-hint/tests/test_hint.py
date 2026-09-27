#!/usr/bin/env python3
###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
###############################################################################

import math
import subprocess
import unittest
from pathlib import Path

from kernel_sched_hint.devices import device
from kernel_sched_hint.fit import fit_weights
from kernel_sched_hint.hint import predict_metadata, predict_text
from kernel_sched_hint.metadata import KernelMetadata, Argument, parse_amdgpu_metadata
from kernel_sched_hint.model import KernelWork, evaluate, predict_work

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GRID = (1048576, 1, 1)
BLOCK = (256, 1, 1)


def _text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


class MetadataTests(unittest.TestCase):
    def test_saxpy_record_matches_comgr_fields(self):
        metas = parse_amdgpu_metadata(_text("saxpy.s"))
        self.assertEqual(len(metas), 1)
        meta = metas[0]
        self.assertEqual(meta.name, "saxpy")
        self.assertEqual(meta.vgprs, 4)
        self.assertEqual(meta.sgprs, 8)
        self.assertEqual(meta.wavefront_size, 64)
        self.assertEqual(meta.kernarg_segment_size, 24)
        buffers = meta.global_buffers()
        self.assertEqual(len(buffers), 2)
        self.assertEqual(len(meta.arguments), 4)
        self.assertTrue(meta.arguments[-1].hidden)
        self.assertEqual(meta.arguments[2].value_kind, "by_value")


class PredictTests(unittest.TestCase):
    def test_saxpy_is_memory_bound_and_uses_metadata(self):
        pred = predict_text(_text("saxpy.s"), "gfx942", GRID, BLOCK)[0]
        self.assertEqual(pred.bound, "memory")
        self.assertEqual(pred.resource, "hbm")
        self.assertEqual(pred.overlap_with, "alu")
        self.assertEqual(pred.waves, 1048576 * 256 // 64)
        self.assertGreater(pred.duration_s, pred.roofline_s)
        self.assertEqual(pred.class_confidence, "high")
        self.assertEqual(pred.time_confidence, "low")
        self.assertTrue(any("4 VGPRs" in note for note in pred.notes))

    def test_ir_saxpy_matches_asm_bound(self):
        asm = predict_text(_text("saxpy.s"), "mi300x", GRID, BLOCK)[0]
        ir = predict_text(_text("saxpy.ll"), "mi300x", GRID, BLOCK)[0]
        self.assertEqual(ir.bound, "memory")
        self.assertEqual(asm.bound, ir.bound)
        # IR does fmul+fadd (2 flops) and three dword accesses, same as the asm.
        self.assertAlmostEqual(ir.flops, asm.flops)
        self.assertAlmostEqual(ir.bytes, asm.bytes)

    def test_fma_chain_is_alu(self):
        pred = predict_text(_text("fma_chain.s"), "gfx942", GRID, BLOCK)[0]
        self.assertEqual(pred.bound, "alu")
        self.assertEqual(pred.resource, "valu")
        self.assertEqual(pred.overlap_with, "memory")

    def test_mfma_uses_matrix_pipe(self):
        pred = predict_text(_text("mfma.s"), "gfx942", GRID, BLOCK)[0]
        self.assertEqual(pred.bound, "alu")
        self.assertEqual(pred.resource, "matrix")

    def test_short_waitcnt_chain_is_latency(self):
        pred = predict_text(_text("latency.s"), "gfx942", (1, 1, 1), (64, 1, 1))[0]
        self.assertEqual(pred.bound, "latency")
        self.assertEqual(pred.overlap_with, "either")

    def test_trip_count_scales_duration_not_bound(self):
        once = predict_text(
            _text("fma_chain.s"),
            "gfx942",
            GRID,
            BLOCK,
            trip_count=1,
            trip_count_known=True,
        )[0]
        twice = predict_text(
            _text("fma_chain.s"),
            "gfx942",
            GRID,
            BLOCK,
            trip_count=2,
            trip_count_known=True,
        )[0]
        self.assertEqual(once.bound, twice.bound)
        self.assertGreater(twice.duration_s, once.duration_s * 1.9)

    def test_metadata_only_does_not_invent_a_high_confidence_mix(self):
        meta = KernelMetadata(
            name="copy",
            vgprs=32,
            sgprs=16,
            wavefront_size=64,
            arguments=[
                Argument(
                    "global_buffer", size=8, address_space="global", access="read_only"
                ),
                Argument(
                    "global_buffer", size=8, address_space="global", access="write_only"
                ),
                Argument("hidden_global_offset_x", size=8, hidden=True),
            ],
        )
        pred = predict_metadata(meta, "gfx942", GRID, BLOCK)
        self.assertEqual(pred.bound, "memory")
        self.assertEqual(pred.class_confidence, "low")
        self.assertTrue(
            any("Instruction mix was not counted" in note for note in pred.notes)
        )
        self.assertEqual(len(meta.global_buffers()), 2)

    def test_isa_mix_wins_over_argument_kinds(self):
        # Two global buffers would look like memory if we only had metadata.
        # A long FMA body with no loads is still ALU.
        text = """
alu_body:
  v_fma_f32 v0, v0, v0, v0
  v_fma_f32 v0, v0, v0, v0
  v_fma_f32 v0, v0, v0, v0
  v_fma_f32 v0, v0, v0, v0
  s_endpgm
.amdgpu_metadata
---
amdhsa.kernels:
  - .name: alu_body
    .vgpr_count: 8
    .sgpr_count: 8
    .wavefront_size: 64
    .args:
      - .size: 8
        .value_kind: global_buffer
        .address_space: global
        .access: read_only
      - .size: 8
        .value_kind: global_buffer
        .address_space: global
        .access: write_only
...
.end_amdgpu_metadata
"""
        # Repeat the fmas enough that the pipe beats a couple of invented loads.
        text = text.replace(
            "  v_fma_f32 v0, v0, v0, v0\n", "  v_fma_f32 v0, v0, v0, v0\n" * 100
        )
        pred = predict_text(text, "gfx942", GRID, BLOCK)[0]
        self.assertEqual(pred.bound, "alu")
        self.assertEqual(pred.resource, "valu")

    def test_private_segment_is_scratch_traffic(self):
        meta = KernelMetadata(
            name="spill",
            vgprs=128,
            sgprs=32,
            wavefront_size=64,
            private_segment_fixed_size=64,
            vgpr_spill_count=8,
        )
        pred = predict_metadata(meta, "gfx942", GRID, BLOCK)
        self.assertEqual(pred.bound, "memory")
        self.assertGreater(pred.bytes, 0)
        self.assertTrue(any("Spills" in note for note in pred.notes))

    def test_lds_and_sgpr_shrink_occupancy(self):
        dev = device("gfx942")
        bare = KernelWork(vgprs=32, wavefront_size=64, valu_f32_flops=1)
        wide = evaluate(bare, dev, waves=1, passes=1)
        capped = evaluate(
            KernelWork(vgprs=32, sgprs=200, wavefront_size=64, valu_f32_flops=1),
            dev,
            waves=1,
            passes=1,
            lds_bytes_per_workgroup=32 * 1024,
            workgroup_size=256,
        )
        self.assertEqual(wide.resident_waves, 304 * 4 * 8)
        self.assertLess(capped.resident_waves, wide.resident_waves)
        self.assertEqual(capped.resident_waves, 304 * 2 * 4)

    def test_buffer_size_caps_global_traffic(self):
        meta = KernelMetadata(
            name="tiny",
            vgprs=16,
            sgprs=8,
            wavefront_size=64,
            isa_size=64,
            arguments=[Argument("global_buffer", size=8)],
            global_buffer_bytes=[128],
        )
        pred = predict_metadata(meta, "gfx942", GRID, BLOCK)
        self.assertLessEqual(pred.bytes, 128)

    def test_fit_recovers_a_constant_scale(self):
        dev = device("gfx942")
        rows = []
        for flops, nbytes, waves in (
            (1.0e5, 1.0e4, 200000),
            (2.0e5, 5.0e4, 400000),
            (8.0e4, 2.0e4, 200000),
            (3.0e5, 1.0e3, 400000),
        ):
            work = KernelWork(
                valu_f32_flops=flops,
                global_bytes=nbytes,
                valu_issues=10,
                mem_ops=1,
                vgprs=32,
                wavefront_size=64,
            )
            terms = evaluate(work, dev, waves, 1)
            measured = terms.t_throughput * 5.0 + terms.t_latency + terms.t_launch
            rows.append((work, dev, waves, 1, measured))
        weights = fit_weights(rows)
        for work, dev, waves, trip, measured in rows:
            pred = predict_work(work, dev, waves, trip_count=trip, weights=weights)
            self.assertAlmostEqual(pred.duration_s, measured, delta=measured * 0.02)

    def test_cpp_matches_python(self):
        work = KernelWork(
            valu_f32_flops=128,
            global_bytes=768,
            mem_ops=3,
            waitcnts=2,
            valu_issues=2,
            vgprs=4,
            sgprs=8,
            wavefront_size=64,
        )
        pred = predict_work(work, device("gfx942"), 4194304)
        binary = Path("/tmp/kernel_sched_hint_test")
        source = Path(__file__).resolve().parents[1] / "cxx" / "test_hint.cpp"
        include = source.parent
        compile = subprocess.run(
            ["g++", "-std=c++17", "-I", str(include), str(source), "-o", str(binary)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(compile.returncode, 0, compile.stderr)
        ran = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
        bound, resource, roofline, duration = ran.stdout.splitlines()
        self.assertEqual(bound, pred.bound)
        self.assertEqual(resource, pred.resource)
        self.assertTrue(
            math.isclose(float(roofline), pred.roofline_s, rel_tol=1e-9, abs_tol=1e-12)
        )
        self.assertTrue(
            math.isclose(float(duration), pred.duration_s, rel_tol=1e-9, abs_tol=1e-12)
        )


if __name__ == "__main__":
    unittest.main()
