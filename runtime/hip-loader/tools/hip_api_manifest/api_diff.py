#!/usr/bin/env python3
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


STATUS_ORDER = {
    "UNCHANGED": 0,
    "MINOR": 1,
    "MAJOR": 2,
}


@dataclass(frozen=True)
class ApiChange:
    status: str
    kind: str
    symbol: str
    detail: str


@dataclass(frozen=True)
class ApiDiff:
    status: str
    changes: tuple[ApiChange, ...]

    def to_jsonable(self) -> dict[str, object]:
        return {
            "status": self.status,
            "changes": [
                {
                    "status": change.status,
                    "kind": change.kind,
                    "symbol": change.symbol,
                    "detail": change.detail,
                }
                for change in self.changes
            ],
        }


def load_manifest(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"manifest root must be an object: {path}")
    return loaded


def diff_manifests(old_manifest: dict[str, object], new_manifest: dict[str, object]) -> ApiDiff:
    old_api = semantic_api(old_manifest)
    new_api = semantic_api(new_manifest)
    changes: list[ApiChange] = []
    compare_functions(old_api, new_api, changes)
    compare_records(old_api, new_api, changes)
    compare_enums(old_api, new_api, changes)
    status = "UNCHANGED"
    for change in changes:
        if STATUS_ORDER[change.status] > STATUS_ORDER[status]:
            status = change.status
    return ApiDiff(status, tuple(changes))


def semantic_api(manifest: dict[str, object]) -> dict[str, object]:
    header_api = manifest.get("header_api")
    if isinstance(header_api, dict):
        return header_api
    return manifest


def compare_functions(
    old_api: dict[str, object],
    new_api: dict[str, object],
    changes: list[ApiChange],
) -> None:
    old_functions = object_dict(old_api, "functions")
    new_functions = object_dict(new_api, "functions")
    for name in sorted(old_functions):
        if name not in new_functions:
            changes.append(ApiChange("MAJOR", "function_removed", name, "function was removed"))
            continue
        old_signature = function_signature(old_functions[name])
        new_signature = function_signature(new_functions[name])
        if old_signature != new_signature:
            changes.append(ApiChange(
                "MAJOR",
                "function_signature_changed",
                name,
                f"{old_signature} -> {new_signature}",
            ))
    for name in sorted(set(new_functions) - set(old_functions)):
        changes.append(ApiChange("MINOR", "function_added", name, "function was added"))


def compare_records(
    old_api: dict[str, object],
    new_api: dict[str, object],
    changes: list[ApiChange],
) -> None:
    old_records = object_dict(old_api, "records")
    new_records = object_dict(new_api, "records")
    for name in sorted(old_records):
        if name not in new_records:
            changes.append(ApiChange("MAJOR", "record_removed", name, "record was removed"))
            continue
        old_fields = field_signature(old_records[name])
        new_fields = field_signature(new_records[name])
        if old_fields != new_fields:
            changes.append(ApiChange(
                "MAJOR",
                "record_layout_changed",
                name,
                f"{old_fields} -> {new_fields}",
            ))
    for name in sorted(set(new_records) - set(old_records)):
        changes.append(ApiChange("MINOR", "record_added", name, "record was added"))


def compare_enums(
    old_api: dict[str, object],
    new_api: dict[str, object],
    changes: list[ApiChange],
) -> None:
    old_enums = object_dict(old_api, "enums")
    new_enums = object_dict(new_api, "enums")
    for name in sorted(old_enums):
        if name not in new_enums:
            changes.append(ApiChange("MAJOR", "enum_removed", name, "enum was removed"))
            continue
        old_constants = enum_signature(old_enums[name])
        new_constants = enum_signature(new_enums[name])
        if old_constants == new_constants:
            continue
        old_names = [constant_name for constant_name, _constant_value in old_constants]
        new_names = [constant_name for constant_name, _constant_value in new_constants]
        if all(name in new_names for name in old_names):
            changed_values = [
                constant_name for constant_name, constant_value in old_constants
                if constant_value is not None
                and dict(new_constants).get(constant_name) not in {None, constant_value}
            ]
            if not changed_values:
                changes.append(ApiChange("MINOR", "enum_constants_added", name, "enum constants were added"))
                continue
        changes.append(ApiChange(
            "MAJOR",
            "enum_changed",
            name,
            f"{old_constants} -> {new_constants}",
        ))
    for name in sorted(set(new_enums) - set(old_enums)):
        changes.append(ApiChange("MINOR", "enum_added", name, "enum was added"))


def object_dict(parent: dict[str, object], key: str) -> dict[str, object]:
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"manifest key must be an object: {key}")
    result: dict[str, object] = {}
    for item_key, item_value in value.items():
        if not isinstance(item_key, str):
            raise ValueError(f"manifest key contains a non-string entry: {key}")
        result[item_key] = item_value
    return result


def function_signature(value: object) -> tuple[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise ValueError("function manifest entry must be an object")
    return_type = value.get("return_type")
    if not isinstance(return_type, str):
        raise ValueError("function manifest entry requires return_type")
    params = value.get("params", [])
    if not isinstance(params, list):
        raise ValueError("function params must be a list")
    param_types: list[str] = []
    for param in params:
        if not isinstance(param, dict):
            raise ValueError("function param must be an object")
        param_type = param.get("type")
        if not isinstance(param_type, str):
            raise ValueError("function param requires type")
        param_types.append(param_type)
    return return_type, tuple(param_types)


def field_signature(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict):
        raise ValueError("record manifest entry must be an object")
    fields = value.get("fields", [])
    if not isinstance(fields, list):
        raise ValueError("record fields must be a list")
    result: list[tuple[str, str]] = []
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("record field must be an object")
        name = field.get("name")
        field_type = field.get("type")
        if not isinstance(name, str) or not isinstance(field_type, str):
            raise ValueError("record field requires name and type")
        result.append((name, field_type))
    return tuple(result)


def enum_signature(value: object) -> tuple[tuple[str, str | None], ...]:
    if not isinstance(value, dict):
        raise ValueError("enum manifest entry must be an object")
    constants = value.get("constants", [])
    if not isinstance(constants, list):
        raise ValueError("enum constants must be a list")
    result: list[tuple[str, str | None]] = []
    for constant in constants:
        if isinstance(constant, list) and len(constant) == 2:
            name, enum_value = constant
        elif isinstance(constant, dict):
            name = constant.get("name")
            enum_value = constant.get("value")
        else:
            raise ValueError("enum constant must be a two-element list or object")
        if not isinstance(name, str):
            raise ValueError("enum constant requires name")
        if enum_value is not None and not isinstance(enum_value, str):
            enum_value = str(enum_value)
        result.append((name, enum_value))
    return tuple(result)


def render_diff(diff: ApiDiff) -> str:
    lines = [diff.status]
    for change in diff.changes:
        lines.append(f"- {change.status} {change.kind} {change.symbol}: {change.detail}")
    return "\n".join(lines) + "\n"
