# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Load metric library YAML files with file-level inheritance.

A file names its parent with a top-level "inherit: <path>" key, relative to the file.
The child is deep-merged onto the parent:
- mappings merge key by key, so a child metric lists only the fields that change;
- scalars and lists replace the parent's value;
- "key: !remove" deletes the key.

Duplicate keys and removing a missing key are errors. Uses the vendored PyYAML, so it
is safe in profile mode.
"""

import copy
from pathlib import Path
from typing import Any, Optional

from vendored import yaml

INHERIT_KEY = "inherit"
REMOVE_TAG = "!remove"


def load_inherited_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file merged with its "inherit" chain.

    Args:
        path: The file to load.

    Returns:
        The merged document, without the "inherit" key.
    """
    chain: dict[Path, dict[str, Any]] = {}  # The file first, its base last.
    next_path: Optional[Path] = Path(path).resolve()
    while next_path is not None:
        if next_path in chain:
            names = [str(chain_path) for chain_path in [*chain, next_path]]
            raise InheritanceError("inheritance cycle: " + " -> ".join(names))
        chain[next_path] = document = _read_document(next_path)
        next_path = _parent_path(next_path, document.pop(INHERIT_KEY, None))
    merged: dict[str, Any] = {}
    for chain_path, document in reversed(chain.items()):
        merged = _merge_mappings(merged, document, str(chain_path))
    return merged


def _parent_path(path: Path, parent: object) -> Optional[Path]:
    if parent is None:
        return None
    if not isinstance(parent, str) or not parent or Path(parent).is_absolute():
        raise InheritanceError(f"{path}: '{INHERIT_KEY}' must be a relative file path")
    parent_path = (path.parent / parent).resolve()
    if not parent_path.is_file():
        raise InheritanceError(f"{path}: parent file not found: {parent_path}")
    return parent_path


def _read_document(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as stream:
            document = yaml.load(stream, Loader=InheritLoader)
    except InheritanceError as error:
        raise InheritanceError(f"{path}: {error}") from error
    if not isinstance(document, dict):
        raise InheritanceError(f"{path}: the top level must be a mapping")
    return document


def _merge_mappings(
    base: dict[str, Any], override: dict[str, Any], source: str
) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Remove):
            _remove_key(merged, key, source)
            continue
        merged[key] = _merge_value(merged.get(key), value, key, source)
    return merged


def _remove_key(mapping: dict[str, Any], key: str, source: str) -> None:
    if key not in mapping:
        raise InheritanceError(
            f"{source}: cannot remove '{key}': the parent has no such key"
        )
    del mapping[key]


def _merge_value(base: object, override: object, key: str, source: str) -> object:
    if isinstance(override, dict):
        return _merge_mappings(base if isinstance(base, dict) else {}, override, source)
    if isinstance(override, list) and _contains_remove(override):
        raise InheritanceError(f"{source}: '{REMOVE_TAG}' is not allowed in '{key}'")
    return copy.deepcopy(override)


def _contains_remove(value: object) -> bool:
    if isinstance(value, Remove):
        return True
    if isinstance(value, dict):
        return any(_contains_remove(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_remove(item) for item in value)
    return False


def _construct_remove(loader: "InheritLoader", node: yaml.Node) -> "Remove":
    if not isinstance(node, yaml.ScalarNode) or loader.construct_scalar(node):
        raise InheritanceError(
            f"line {node.start_mark.line + 1}: '{REMOVE_TAG}' takes no value"
        )
    return Remove()


class Remove:
    """Marks a key to delete during the merge."""

    def __repr__(self) -> str:
        return "Remove()"


class InheritLoader(yaml.SafeLoader):
    """SafeLoader with the "!remove" tag that rejects duplicate keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
        if duplicates:
            raise InheritanceError(
                f"line {node.start_mark.line + 1}: duplicate keys {duplicates}"
            )
        return super().construct_mapping(node, deep=deep)


InheritLoader.add_constructor(REMOVE_TAG, _construct_remove)


class InheritanceError(Exception):
    """A metric library file cannot be loaded or merged.

    The loader raises instead of logging so that each caller decides what to do: the
    validator collects the errors of every file, and the CLI reports one with
    console_error().
    """
