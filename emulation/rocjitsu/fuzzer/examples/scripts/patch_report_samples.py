def compact_kernel_name(name, limit=72):
    name = str(name)
    if len(name) <= limit:
        return name
    head = max(8, (limit - 3) // 2)
    tail = max(8, limit - 3 - head)
    return f"{name[:head]}...{name[-tail:]}"


def selected_edge_key(edge):
    return (
        edge.get("kernel", ""),
        edge.get("kind", ""),
        edge.get("patch_text_offset", 0),
        edge.get("slot_policy", ""),
        edge.get("fixed_slot", 0),
        bool(edge.get("scratch_spill")),
        bool(edge.get("vgpr_scratch_spill")),
        bool(edge.get("sgpr_scratch_spill")),
        edge.get("scratch_address_exec_source", ""),
        edge.get("scratch_address_vgpr", ""),
        tuple(edge.get("scratch_spilled_vgprs") or []),
        tuple(edge.get("scratch_spilled_sgprs") or []),
        edge.get("placement", ""),
    )


def selected_edge_detail(edge):
    kernel = compact_kernel_name(edge.get("kernel", ""))
    kind = edge.get("kind", "")
    policy = edge.get("slot_policy", "")
    patch = edge.get("patch_text_offset", 0)
    fixed_slot = edge.get("fixed_slot", 0)
    scratch_parts = []
    if edge.get("vgpr_scratch_spill"):
        scratch_parts.append("vgpr")
    if edge.get("sgpr_scratch_spill"):
        scratch_parts.append("sgpr")
    if edge.get("scratch_spill") and not scratch_parts:
        scratch_parts.append("scratch")
    scratch = "+".join(scratch_parts) if scratch_parts else "none"
    exec_source = edge.get("scratch_address_exec_source", "")
    placement = edge.get("placement", "")
    detail = f"`{kernel}` {kind}@0x{patch:x} {policy}"
    if fixed_slot:
        detail += f" slot={fixed_slot}"
    if scratch != "none":
        detail += f" scratch={scratch}"
    if exec_source and exec_source != "none":
        detail += f" exec={exec_source}"
    if edge.get("scratch_spill") and "scratch_address_vgpr" in edge:
        detail += f" addr=v{edge.get('scratch_address_vgpr')}"
    spilled_vgprs = edge.get("scratch_spilled_vgprs") or []
    if spilled_vgprs:
        detail += " spill_vgpr=" + ",".join(f"v{reg}" for reg in spilled_vgprs)
    spilled_sgprs = edge.get("scratch_spilled_sgprs") or []
    if spilled_sgprs:
        detail += " spill_sgpr=" + ",".join(f"s{reg}" for reg in spilled_sgprs)
    if placement:
        detail += f" cave={placement}"
    if "trampoline_bytes" in edge:
        detail += f" bytes={edge.get('trampoline_bytes')}"
    return detail


def selected_edge_sample_text(summaries, limit=4):
    edges = []
    seen = set()
    for summary in summaries:
        for edge in summary.get("patches", {}).get("sampled_selected_edges", []):
            if not isinstance(edge, dict):
                continue
            key = selected_edge_key(edge)
            if key in seen:
                continue
            seen.add(key)
            edges.append(edge)
    edges.sort(key=selected_edge_key)
    if len(edges) > limit:
        selected_edges = edges[:limit]
        if not any(edge.get("scratch_spill") for edge in selected_edges):
            scratch_edge = next((edge for edge in edges if edge.get("scratch_spill")), None)
            if scratch_edge is not None:
                selected_edges[-1] = scratch_edge
    else:
        selected_edges = edges
    samples = [selected_edge_detail(edge) for edge in selected_edges]
    if not samples:
        return ""
    text = ", ".join(samples)
    if len(edges) > len(samples):
        text += ", ..."
    return text


def sum_summary_field(summaries, section, field):
    return sum(summary.get(section, {}).get(field, 0) for summary in summaries)


def coverage_mix_from_summaries(summaries):
    summaries = list(summaries)
    edge_sites = sum_summary_field(summaries, "patches", "edge_sites_patched")
    hashed = sum_summary_field(summaries, "patches", "hashed_edge_sites")
    fixed = sum_summary_field(summaries, "patches", "fixed_edge_sites")
    degraded = sum_summary_field(summaries, "patches", "branch_edges_degraded_to_fixed")
    if edge_sites == 0:
        return "none"
    if hashed > 0 and fixed > 0:
        if degraded > 0:
            return "previous-BB + fixed fallback"
        return "previous-BB + fixed"
    if hashed > 0:
        return "previous-BB"
    if fixed > 0:
        return "fixed branch counters"
    return "instrumented, unclassified"


def coverage_mix_from_reports(reports):
    return coverage_mix_from_summaries(report["summary"] for report in reports)
