# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Load and check the per-runtime configuration (YAML)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import yaml

# Keyword names a configuration may add to an API's events, beyond API, which
# every probe event carries. They are the suffixes of the ROCM_TRACE_KW_*
# macros in rocm/trace/values.h; a test keeps the two lists in step.
EXTRA_KEYWORDS = ("DISPATCH", "STATE", "SERIALIZE", "COPY_TIME", "CODE_BYTES", "ALLOC")
LEVELS = ("information", "verbose")
RESULT = "result"

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SIZE_MEMBER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_INCLUDE = re.compile(r"^[A-Za-z0-9_./+-]+$")


class ConfigError(Exception):
    """The configuration file is malformed."""


@dataclass(frozen=True)
class TableConfig:
    name: str
    size_member: str


@dataclass(frozen=True)
class ApiConfig:
    """Overrides for one API. Names are field names: a parameter's name, or
    ``arg_<name>`` for one RC-API-1 prefixes (``result``, ``tid``,
    ``timestamp``, ``exception``, ``arg_*``), or ``arg_<N>`` for an unnamed
    one; ``result`` is the return value."""

    out: Tuple[str, ...] = ()
    strings: Tuple[str, ...] = ()
    uncapped: Tuple[str, ...] = ()
    omit: Tuple[str, ...] = ()
    level: str = ""
    keywords: Tuple[str, ...] = ()
    skip: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Config:
    path: Path
    provider: str
    output_name: str
    includes: Tuple[str, ...]
    tables: Tuple[TableConfig, ...]
    success: Mapping[str, str]
    level: str
    capture_level: str = ""
    apis: Mapping[str, ApiConfig] = field(default_factory=dict)


class _UniqueKeyLoader(yaml.SafeLoader):
    """A SafeLoader that rejects duplicate mapping keys."""


def _construct_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ConfigError(
                f"line {key_node.start_mark.line + 1}: duplicate key '{key}'"
            )
        seen.add(key)
    return loader.construct_mapping(node, deep=deep)


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _keys(
    where: str,
    data: Mapping[str, Any],
    required: Tuple[str, ...],
    optional: Tuple[str, ...],
) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise ConfigError(f"{where}: missing {', '.join(missing)}")
    unknown = sorted(set(data) - set(required) - set(optional))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {', '.join(unknown)}")


def _identifier(where: str, value: Any) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.match(value):
        raise ConfigError(f"{where}: expected an identifier, got {value!r}")
    return value


def _names(where: str, value: Any) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise ConfigError(f"{where}: expected a list of names")
    names = tuple(_identifier(where, v) for v in value)
    if len(set(names)) != len(names):
        raise ConfigError(f"{where}: repeated name")
    return names


def _level(where: str, value: Any) -> str:
    if value not in LEVELS:
        raise ConfigError(f"{where}: level must be one of {', '.join(LEVELS)}")
    return value


def _api(name: str, data: Any) -> ApiConfig:
    where = f"apis.{name}"
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping")
    _keys(
        where,
        data,
        (),
        ("out", "strings", "uncapped", "omit", "level", "keywords", "skip", "reason"),
    )

    if data.get("skip", False):
        if data["skip"] is not True:
            raise ConfigError(f"{where}.skip: expected true")
        reason = data.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ConfigError(f"{where}: skip needs a reason")
        others = sorted(set(data) - {"skip", "reason"})
        if others:
            raise ConfigError(f"{where}: a skipped API takes no {', '.join(others)}")
        return ApiConfig(skip=True, reason=reason.strip())
    if "reason" in data:
        raise ConfigError(f"{where}: reason is only for skip")

    out = _names(f"{where}.out", data.get("out", []))
    strings = _names(f"{where}.strings", data.get("strings", []))
    uncapped = _names(f"{where}.uncapped", data.get("uncapped", []))
    omit = _names(f"{where}.omit", data.get("omit", []))
    keywords = _names(f"{where}.keywords", data.get("keywords", []))
    if RESULT in out:
        raise ConfigError(
            f"{where}.out: result is the return value, always recorded; "
            "a parameter named result is arg_result"
        )
    both = sorted(set(omit) & (set(out) | set(strings) | set(uncapped)))
    if both:
        raise ConfigError(f"{where}: {', '.join(both)} both omitted and recorded")
    unknown = sorted(set(keywords) - set(EXTRA_KEYWORDS))
    if unknown:
        raise ConfigError(
            f"{where}.keywords: unknown {', '.join(unknown)}; "
            f"choose from {', '.join(EXTRA_KEYWORDS)}"
        )
    level = _level(f"{where}.level", data["level"]) if "level" in data else ""
    return ApiConfig(out, strings, uncapped, omit, level, keywords)


def load(path: Path) -> Config:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as err:
        raise ConfigError(f"{path}: {err}") from err
    except ConfigError as err:
        raise ConfigError(f"{path}: {err}") from err
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    try:
        return _config(path, data)
    except ConfigError as err:
        raise ConfigError(f"{path}: {err}") from err


def _config(path: Path, data: Dict[str, Any]) -> Config:
    _keys(
        "top level",
        data,
        ("version", "provider", "output_name", "includes", "tables"),
        ("success", "defaults", "capture_level", "apis"),
    )
    if data["version"] != 1:
        raise ConfigError("version: only version 1 is supported")

    includes = data["includes"]
    if not isinstance(includes, list) or not includes:
        raise ConfigError("includes: expected a non-empty list of header paths")
    for inc in includes:
        if not isinstance(inc, str) or not _INCLUDE.match(inc):
            raise ConfigError(f"includes: not a header path: {inc!r}")

    tables_data = data["tables"]
    if not isinstance(tables_data, list) or not tables_data:
        raise ConfigError("tables: expected a non-empty list")
    tables: List[TableConfig] = []
    for i, t in enumerate(tables_data):
        where = f"tables[{i}]"
        if not isinstance(t, dict):
            raise ConfigError(f"{where}: expected a mapping")
        _keys(where, t, ("name", "size_member"), ())
        size_member = t["size_member"]
        if not isinstance(size_member, str) or not _SIZE_MEMBER.match(size_member):
            raise ConfigError(f"{where}.size_member: expected a member name")
        tables.append(TableConfig(_identifier(f"{where}.name", t["name"]), size_member))
    if len({t.name for t in tables}) != len(tables):
        raise ConfigError("tables: a table is listed twice")

    success = data.get("success", {})
    if not isinstance(success, dict):
        raise ConfigError("success: expected a mapping of result type to success value")
    for k, v in success.items():
        _identifier("success", k)
        _identifier(f"success.{k}", v)

    defaults = data.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ConfigError("defaults: expected a mapping")
    _keys("defaults", defaults, (), ("level",))
    level = _level("defaults.level", defaults.get("level", "information"))

    apis_data = data.get("apis", {}) or {}
    if not isinstance(apis_data, dict):
        raise ConfigError("apis: expected a mapping")
    apis = {_identifier("apis", k): _api(k, v) for k, v in apis_data.items()}

    return Config(
        path=path,
        provider=_identifier("provider", data["provider"]),
        output_name=_identifier("output_name", data["output_name"]),
        includes=tuple(includes),
        tables=tuple(tables),
        success=dict(success),
        level=level,
        capture_level=(
            _identifier("capture_level", data["capture_level"])
            if "capture_level" in data
            else ""
        ),
        apis=apis,
    )
