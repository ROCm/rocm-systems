# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Cross-architecture consistency tests for memory chart layouts."""

from tests.unit.memory_chart.conftest import load_layout_json


def _block_ids(layout: dict) -> set[str]:
    return {b["id"] for b in layout["blocks"]}


class TestCdnaFamilyConsistency:
    """Verify incremental topology relationships among CDNA layouts."""

    def test_gfx94x_is_gfx90x_plus_mall(self) -> None:
        gfx90x = load_layout_json("gfx90x")
        gfx94x = load_layout_json("gfx94x")

        ids_90x = _block_ids(gfx90x)
        ids_94x = _block_ids(gfx94x)

        added = ids_94x - ids_90x
        assert "mall" in added, "gfx94x should add the MALL block"
        assert ids_90x.issubset(ids_94x), (
            f"gfx90x blocks {ids_90x - ids_94x} missing in gfx94x"
        )

    def test_gfx950_is_gfx94x_plus_io(self) -> None:
        gfx94x = load_layout_json("gfx94x")
        gfx950 = load_layout_json("gfx950")

        ids_94x = _block_ids(gfx94x)
        ids_950 = _block_ids(gfx950)

        added = ids_950 - ids_94x
        assert "pcie" in added, "gfx950 should add the PCIe block"
        assert ids_94x.issubset(ids_950), (
            f"gfx94x blocks {ids_94x - ids_950} missing in gfx950"
        )

    def test_gfx950_has_lds_rwa_arrows(self) -> None:
        gfx950 = load_layout_json("gfx950")
        lds_arrows = [
            a for a in gfx950["arrows"] if a["to"] == "lds" and a["metric"] != "LDS Req"
        ]
        lds_metrics = {a["metric"] for a in lds_arrows}
        assert {"LDS Read", "LDS Write", "LDS Atomic"}.issubset(lds_metrics)

    def test_gfx94x_has_estimated_hbm_bw(self) -> None:
        gfx94x = load_layout_json("gfx94x")
        hbm_block = next(b for b in gfx94x["blocks"] if b["id"] == "hbm")
        hbm_metrics = {c["metric"] for c in hbm_block["content"]}
        assert "Estimated HBM Read BW" in hbm_metrics
        assert "Estimated HBM Write and Atomic BW" in hbm_metrics

    def test_gfx90x_hbm_is_empty(self) -> None:
        gfx90x = load_layout_json("gfx90x")
        hbm_block = next(b for b in gfx90x["blocks"] if b["id"] == "hbm")
        assert hbm_block["content"] == []

    def test_all_cdna_share_cu_block_content(self) -> None:
        cu_contents = []
        for name in ("gfx90x", "gfx94x", "gfx950"):
            layout = load_layout_json(name)
            cu_block = next(b for b in layout["blocks"] if b["id"] == "cu")
            cu_metrics = tuple(c["metric"] for c in cu_block["content"])
            cu_contents.append((name, cu_metrics))
        first_name, first_metrics = cu_contents[0]
        for other_name, other_metrics in cu_contents[1:]:
            assert first_metrics == other_metrics, (
                f"CU metrics differ: {first_name} vs {other_name}"
            )


class TestGfx1250NestingStructure:
    def test_tcp_has_lds_and_gl0_children(self) -> None:
        layout = load_layout_json("gfx1250")
        tcp_block = next(b for b in layout["blocks"] if b["id"] == "tcp")
        assert set(tcp_block["children"]) == {"lds", "gl0"}

    def test_sqc_has_icache_and_dcache_children(self) -> None:
        layout = load_layout_json("gfx1250")
        sqc_block = next(b for b in layout["blocks"] if b["id"] == "sqc")
        assert set(sqc_block["children"]) == {
            "icache",
            "dcache",
        }
