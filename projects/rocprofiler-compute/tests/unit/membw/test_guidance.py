# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

from membw.guidance import render_guidance_blocks


class TestRenderGuidanceBlocks:
    def test_replaces_metric_and_threshold_placeholders(self):
        templates = {
            "t1": "Measured: {metric:m1}% (threshold: {threshold:th}%)",
        }
        blocks = render_guidance_blocks(
            ["t1"],
            templates,
            thresholds={"th": 10.0},
            metric_values={"m1": 22.4},
        )
        assert "22.4%" in blocks[0]
        assert "10%" in blocks[0]

    def test_none_metric_renders_na(self):
        templates = {"t1": "Value: {metric:m1}"}
        blocks = render_guidance_blocks(["t1"], templates, {}, {"m1": None})
        assert "N/A" in blocks[0]

    def test_overflow_capping(self):
        templates = {f"t{i}": f"Block {i}" for i in range(8)}
        blocks = render_guidance_blocks(
            [f"t{i}" for i in range(8)],
            templates,
            {},
            {},
            max_blocks=3,
        )
        assert len(blocks) == 4
        assert "...and 5 more" in blocks[3]

    def test_empty_ids(self):
        assert render_guidance_blocks([], {}, {}, {}) == ()

    def test_missing_template_skipped(self):
        assert render_guidance_blocks(["missing"], {}, {}, {}) == ()

    def test_template_error_includes_exception_info(self):
        templates = {"t1": "Threshold: {threshold:th}%"}
        blocks = render_guidance_blocks(
            ["t1"], templates, thresholds={"th": "not_a_number"}, metric_values={}
        )
        assert len(blocks) == 1
        assert "template error" in blocks[0]
        assert "Error" in blocks[0] or "TypeError" in blocks[0]
