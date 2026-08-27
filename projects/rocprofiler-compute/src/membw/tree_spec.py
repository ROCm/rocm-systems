# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load and validate the memory bandwidth bottleneck tree specification."""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import config
from membw.models import NodeSpec, TreeSpec
from utils.logger import console_error
from utils.utils_common import load_yaml

# Duplicated from engine._OPS keys to avoid circular import (engine imports tree_spec).
_VALID_OPS = frozenset({"gte", "gt", "lt", "lte"})
_VALID_LEVELS = frozenset({"GL1", "GL2", "EA"})

_KNOWN_SCHEMA_HASHES: frozenset[str] = frozenset({"4e164575e9d1fc51"})


def load_tree_spec(arch: str) -> TreeSpec:
    """Load and validate the tree spec for an architecture."""
    spec_path = tree_spec_path(arch)
    if not spec_path.exists():
        console_error("membw", f"No tree spec for arch {arch!r}: {spec_path}")
    raw = load_yaml(spec_path)
    if not isinstance(raw, dict):
        console_error("membw", f"Tree spec must be a YAML mapping: {spec_path}")

    guidance_path = _guidance_path(arch)
    guidance_templates: dict[str, str] = {}
    if guidance_path.exists():
        guidance_raw = load_yaml(guidance_path)
        if isinstance(guidance_raw, dict):
            guidance_templates = guidance_raw.get("guidance_templates", {})

    spec = _parse_tree_spec(raw, guidance_templates)
    _validate_tree_spec(spec)
    return spec


def collect_metric_keys(spec: TreeSpec) -> frozenset[str]:
    """Collect all unique metric keys referenced by the tree."""
    keys: set[str] = set()
    for root in spec.roots:
        _collect_keys_recursive(root, keys)
    return frozenset(keys)


# --- Private helpers ---


def tree_spec_path(arch: str) -> Path:
    """Return the path to the tree spec YAML for an architecture."""
    return (
        config.rocprof_compute_home
        / "membw"
        / "tree_spec"
        / f"{arch}_membw_tree_spec.yaml"
    )


def _guidance_path(arch: str) -> Path:
    """Return the path to the guidance YAML for an architecture."""
    return (
        config.rocprof_compute_home
        / "membw"
        / "tree_spec"
        / f"{arch}_membw_guidance.yaml"
    )


def _parse_tree_spec(
    raw: dict[str, Any],
    external_guidance: Optional[dict[str, str]] = None,
) -> TreeSpec:
    """Parse raw YAML dict into a TreeSpec."""
    thresholds = raw.get("thresholds", {})
    if not isinstance(thresholds, dict):
        console_error("membw", "'thresholds' must be a mapping")

    raw_nodes = raw.get("nodes", {})
    if not isinstance(raw_nodes, dict):
        console_error("membw", "'nodes' must be a mapping")

    if external_guidance is not None:
        guidance_templates = external_guidance
    else:
        guidance_templates = raw.get("guidance_templates", {})
    if not isinstance(guidance_templates, dict):
        console_error("membw", "'guidance_templates' must be a mapping")

    roots = _parse_nodes(raw_nodes, parent_level=None)
    schema_hash = _compute_schema_hash(raw)

    return TreeSpec(
        thresholds=thresholds,
        roots=roots,
        guidance_templates=guidance_templates,
        schema_hash=schema_hash,
    )


def _parse_nodes(
    raw_nodes: dict[str, Any],
    parent_level: Optional[str],
) -> tuple[NodeSpec, ...]:
    """Recursively parse node dicts into NodeSpec tuples."""
    result = []
    for node_id, node_dict in raw_nodes.items():
        if not isinstance(node_dict, dict):
            console_error("membw", f"Node {node_id!r} must be a mapping")

        level = node_dict.get("level", parent_level)
        raw_children = node_dict.get("children", {})
        if not isinstance(raw_children, dict):
            console_error("membw", f"Node {node_id!r}: 'children' must be a mapping")
        children = _parse_nodes(raw_children, parent_level=level)
        raw_siblings_false = node_dict.get("requires_siblings_false", [])

        result.append(
            NodeSpec(
                id=node_id,
                level=level if level is not None else "",
                metric=node_dict.get("metric"),
                op=node_dict.get("op"),
                threshold_key=node_dict.get("threshold"),
                label=node_dict.get("label", node_id),
                guidance_id=node_dict.get("guidance_id"),
                requires_parent=bool(node_dict.get("requires_parent")),
                requires_siblings_false=tuple(raw_siblings_false),
                children=children,
            )
        )
    return tuple(result)


def _validate_tree_spec(spec: TreeSpec) -> None:
    """Validate internal consistency of a TreeSpec."""
    errors: list[str] = []

    if _KNOWN_SCHEMA_HASHES and spec.schema_hash not in _KNOWN_SCHEMA_HASHES:
        errors.append(
            f"Unknown schema hash {spec.schema_hash!r}. "
            "The tree spec format may have changed."
        )

    for root in spec.roots:
        _validate_node(root, spec, errors, sibling_ids=set())

    if errors:
        console_error(
            "membw",
            "Tree spec validation failed:\n" + "\n".join(f"  - {e}" for e in errors),
        )


def _validate_node(
    node: NodeSpec,
    spec: TreeSpec,
    errors: list[str],
    sibling_ids: set[str],
) -> None:
    """Validate a single node and recurse into children."""
    is_catch_all = node.requires_parent and len(node.requires_siblings_false) > 0

    if node.level not in _VALID_LEVELS:
        errors.append(f"Node {node.id!r}: invalid level {node.level!r}")

    if is_catch_all:
        for sibling_id in node.requires_siblings_false:
            if sibling_id not in sibling_ids:
                errors.append(
                    f"Node {node.id!r}: "
                    f"requires_siblings_false references "
                    f"unknown sibling {sibling_id!r}"
                )
    else:
        if node.metric is None:
            errors.append(f"Node {node.id!r}: non-catch-all node must have a metric")
        if node.op is not None and node.op not in _VALID_OPS:
            errors.append(f"Node {node.id!r}: invalid op {node.op!r}")
        if node.op is None and node.metric is not None:
            errors.append(f"Node {node.id!r}: has metric but no op")
        if node.threshold_key is not None:
            if node.threshold_key not in spec.thresholds:
                errors.append(
                    f"Node {node.id!r}: threshold {node.threshold_key!r} not found"
                )
        elif node.metric is not None:
            errors.append(f"Node {node.id!r}: has metric but no threshold")

    if node.guidance_id is not None:
        if node.guidance_id not in spec.guidance_templates:
            errors.append(
                f"Node {node.id!r}: guidance_id "
                f"{node.guidance_id!r} not found in templates"
            )

    child_ids = {child.id for child in node.children}
    for child in node.children:
        _validate_node(child, spec, errors, sibling_ids=child_ids)


def _collect_keys_recursive(node: NodeSpec, keys: set[str]) -> None:
    """Recursively collect metric keys from a node tree."""
    if node.metric is not None:
        keys.add(node.metric)
    for child in node.children:
        _collect_keys_recursive(child, keys)


def _compute_schema_hash(raw: dict[str, Any]) -> str:
    """Hash the YAML's structural shape (keys and nesting, not values)."""
    shape = _extract_shape(raw)
    shape_json = json.dumps(shape, sort_keys=True)
    return hashlib.sha256(shape_json.encode("utf-8")).hexdigest()[:16]


def _extract_shape(obj: object) -> object:
    """Extract structural shape for schema hashing."""
    if isinstance(obj, dict):
        return [[k, _extract_shape(v)] for k, v in sorted(obj.items())]
    if isinstance(obj, list):
        if not obj:
            return ["list"]
        return ["list", _extract_shape(obj[0])]
    return type(obj).__name__
