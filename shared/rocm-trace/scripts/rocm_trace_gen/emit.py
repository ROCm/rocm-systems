# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Turn parsed tables and a configuration into the probe header and source."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import __version__
from .config import RESULT, ApiConfig, Config
from .fields import STRING_CAP, Field, Unmappable, string, value
from .model import CType, Kind, Param, Slot, Table

_RESULT_VAR = "rocm_trace_result"
_EXCEPTION_VAR = "rocm_trace_exception"
_COPYRIGHT = "// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT\n"
_INDENT = "    "

# Field names a parameter cannot take: RC-API-1 prefixes these with arg_
# (exception is added here, for the exit event's own field).
RESERVED = ("result", "tid", "timestamp", "exception")

# TraceLoggingWrite takes at most 99 arguments; level, keyword and opcode use 3.
MAX_FIELDS = 96


class GenerateError(Exception):
    """The configuration does not fit the tables; one message per problem."""

    def __init__(self, errors: Sequence[str]):
        super().__init__("\n".join(errors))
        self.errors = list(errors)


def field_name(p: Param) -> str:
    """The event field name of a parameter (RC-API-1)."""
    if p.declared and (p.name in RESERVED or p.name.startswith("arg_")):
        return f"arg_{p.name}"
    return p.name


def _tag(
    fields: Iterable[Field],
    param: str,
    param_type: str,
    role: str,
    capture: str,
    on_success: bool = False,
) -> List[Field]:
    """The fields with what the probe schema says about them."""
    return [
        replace(
            f,
            param=param,
            param_type=param_type,
            role=role,
            capture=capture,
            on_success=on_success,
        )
        for f in fields
    ]


def _all(*conditions: Optional[str]) -> Optional[str]:
    parts = [c for c in conditions if c]
    return " && ".join(parts) if parts else None


def _declarable(spelling: str) -> bool:
    """Whether ``<spelling> name`` is a valid declaration."""
    return "(" not in spelling and "[" not in spelling


@dataclass(frozen=True)
class _Probe:
    slot: Slot
    function: str
    exit_function: str
    aliases: Tuple[Tuple[str, str], ...]
    params: Tuple[Tuple[str, str], ...]  # (declared type, name)
    result_type: str
    exit_params: Tuple[Tuple[str, str], ...]  # after the exception flag
    level: str
    keywords: Tuple[str, ...]
    enter: Tuple[Field, ...]
    exit: Tuple[Field, ...]


def _check_names(slot: Slot, api: ApiConfig, by_field: Dict[str, Param]) -> List[str]:
    where = f"{slot.table}.{slot.member} ({slot.api})"
    errors: List[str] = []
    known = set(by_field) | {RESULT}
    for key in ("out", "strings", "uncapped", "omit"):
        for name in getattr(api, key):
            if name not in known:
                hint = ""
                if f"arg_{name}" in by_field:
                    hint = f"; RC-API-1 names this parameter arg_{name}"
                errors.append(
                    f"{where}: {key} names '{name}', which is not a parameter{hint}"
                )
    if errors:
        return errors

    for name in api.out:
        ctype = by_field[name].ctype
        if (
            ctype.kind != Kind.POINTER
            or ctype.pointee is None
            or ctype.pointee.kind == Kind.VOID
        ):
            errors.append(
                f"{where}: out parameter '{name}' is not a pointer to a complete type"
            )
    for name in api.strings:
        if name == RESULT:
            if not slot.result.is_char_pointer:
                errors.append(
                    f"{where}: strings names 'result', which is not a char pointer"
                )
        elif not by_field[name].ctype.is_char_pointer:
            errors.append(
                f"{where}: strings names '{name}', which is not a char-pointer output "
                "or char* input"
            )
        elif name not in api.out and by_field[name].ctype.is_const_char_pointer:
            errors.append(
                f"{where}: strings names '{name}': const char* inputs are recorded as "
                "strings already"
            )
    for name in api.uncapped:
        is_input_string = (
            name in by_field and by_field[name].ctype.is_const_char_pointer
        )
        if name not in api.strings and not (is_input_string and name not in api.out):
            errors.append(
                f"{where}: uncapped names '{name}', which is not a string field"
            )
    return errors


def _plan(
    slot: Slot, api: ApiConfig, config: Config
) -> Tuple[Optional[_Probe], List[str]]:
    where = f"{slot.table}.{slot.member} ({slot.api})"
    by_field = {field_name(p): p for p in slot.params}
    errors = _check_names(slot, api, by_field)
    if errors:
        return None, errors

    cap = config.capture_level
    scalars = f"{cap} >= ROCM_TRACE_CAPTURE_SCALARS" if cap else None
    full = f"{cap} >= ROCM_TRACE_CAPTURE_FULL" if cap else None
    # The lowest capture level that records a field guarded by scalars or full.
    at_scalars = "scalars" if cap else "none"
    at_full = "full" if cap else "none"

    aliases: List[Tuple[str, str]] = []

    def declared_type(spelling: str, alias: str) -> str:
        if _declarable(spelling):
            return spelling
        aliases.append((alias, spelling))
        return alias

    prefix = f"{slot.table}_{slot.api}"
    declared = [
        (declared_type(p.ctype.spelling, f"{prefix}_{p.name}_t"), p.name)
        for p in slot.params
    ]
    result_type = declared_type(slot.result.spelling, f"{prefix}_result_t")

    enter: List[Field] = []
    for p in slot.params:
        name = field_name(p)
        if name in api.omit:
            continue
        try:
            # const char* inputs are strings; a char* input only when listed in strings (an
            # API that is not const-correct), since a char* is usually a buffer.
            is_string = p.ctype.is_const_char_pointer or name in api.strings
            spelling = p.ctype.spelling
            if is_string and name not in api.out:
                text = string(p.name, name, name not in api.uncapped, full, p.ctype)
                enter += _tag([text], p.name, spelling, "argument", at_full)
            else:
                fields = value(p.ctype, p.name, name, scalars)
                enter += _tag(fields, p.name, spelling, "argument", at_scalars)
        except Unmappable as err:
            errors.append(
                f"{where}: parameter '{name}' has no field mapping: {err}; add it to omit"
            )

    has_result = slot.result.kind != Kind.VOID and RESULT not in api.omit
    exit_fields: List[Field] = []
    if has_result:
        try:
            spelling = slot.result.spelling
            if RESULT in api.strings:
                text = string(
                    _RESULT_VAR, RESULT, RESULT not in api.uncapped, full, slot.result
                )
                exit_fields += _tag([text], RESULT, spelling, "result", at_full)
            else:
                fields = value(slot.result, _RESULT_VAR, RESULT)
                exit_fields += _tag(fields, RESULT, spelling, "result", "none")
        except Unmappable as err:
            errors.append(
                f"{where}: the result has no field mapping: {err}; add result to omit"
            )

    success = config.success.get(slot.result.spelling)
    success_cond = f"{_RESULT_VAR} == {success}" if success and has_result else None
    exit_params: List[Tuple[str, str]] = []
    if has_result:
        exit_params.append((result_type, _RESULT_VAR))
    for p, decl in zip(slot.params, declared):
        name = field_name(p)
        if name not in api.out:
            continue
        exit_params.append(decl)
        cond = _all(full, f"!{_EXCEPTION_VAR}", success_cond, f"{p.name} != nullptr")
        try:
            spelling = p.ctype.spelling
            if name in api.strings:
                fields = [string(p.name, name, name not in api.uncapped, cond, p.ctype)]
            else:
                fields = value(p.ctype.pointee, f"(*{p.name})", name, cond)
            exit_fields += _tag(fields, p.name, spelling, "output", at_full, True)
        except Unmappable as err:
            errors.append(
                f"{where}: out parameter '{name}' points to a type with no field "
                f"mapping: {err}"
            )
    exit_fields.append(
        Field(
            "TraceLoggingBoolean",
            (f"static_cast<uint8_t>({_EXCEPTION_VAR})",),
            "exception",
            param="exception",
            param_type="bool",
            role="exception",
        )
    )

    for event, fields in (("enter", enter), ("exit", exit_fields)):
        if len(fields) > MAX_FIELDS:
            errors.append(
                f"{where}: the {event} event would have {len(fields)} fields; "
                f"TraceLogging allows {MAX_FIELDS}"
            )
        seen = set()
        for f in fields:
            if f.name in seen:
                errors.append(
                    f"{where}: the {event} event would have two fields named "
                    f"'{f.name}'; omit one of the parameters"
                )
            seen.add(f.name)

    if errors:
        return None, errors
    return (
        _Probe(
            slot=slot,
            function=f"probe_{prefix}",
            exit_function=f"exit_{prefix}",
            aliases=tuple(aliases),
            params=tuple(declared),
            result_type=result_type,
            exit_params=tuple(exit_params),
            level=api.level or config.level,
            keywords=("API", *api.keywords),
            enter=tuple(enter),
            exit=tuple(exit_fields),
        ),
        [],
    )


def _write(
    provider: str, name: str, probe: _Probe, opcode: str, fields: Sequence[Field]
) -> List[str]:
    keywords = " | ".join(f"ROCM_TRACE_KW_{k}" for k in probe.keywords)
    args = [
        f"TraceLoggingLevel(ROCM_TRACE_LEVEL_{probe.level.upper()})",
        f"TraceLoggingKeyword({keywords})",
        f"TraceLoggingOpcode({opcode})",
        *(f.render() for f in fields),
    ]
    lines = [f'{_INDENT}TraceLoggingWrite({provider}, "{name}",']
    for i, arg in enumerate(args):
        lines.append(f"{_INDENT * 3}{arg}{');' if i == len(args) - 1 else ','}")
    return lines


def _probe_source(probe: _Probe, provider: str, saved: str) -> List[str]:
    slot = probe.slot
    void = slot.result.kind == Kind.VOID
    lines = [f"using {alias} = {spelling};" for alias, spelling in probe.aliases]

    # The one place the exit event is written, so both paths share one layout.
    exit_params = ", ".join(
        [f"bool {_EXCEPTION_VAR}"] + [f"{t} {n}" for t, n in probe.exit_params]
    )
    lines.append(f"void {probe.exit_function}({exit_params}) {{")
    lines += _write(
        provider, f"{slot.api}_exit", probe, "WINEVENT_OPCODE_STOP", probe.exit
    )
    lines += ["}", ""]

    params = ", ".join(f"{t} {n}" for t, n in probe.params)
    call = f"{saved}.{slot.member}({', '.join(n for _, n in probe.params)})"
    exit_args = ", ".join(n for _, n in probe.exit_params)
    exit_args = f", {exit_args}" if exit_args else ""
    lines.append(f"{probe.result_type} {probe.function}({params}) {{")
    lines += _write(
        provider, f"{slot.api}_enter", probe, "WINEVENT_OPCODE_START", probe.enter
    )
    if not void:
        lines.append(f"{_INDENT}{probe.result_type} {_RESULT_VAR}{{}};")
    lines.append(f"{_INDENT}try {{")
    lines.append(
        f"{_INDENT * 2}{call};" if void else f"{_INDENT * 2}{_RESULT_VAR} = {call};"
    )
    lines.append(f"{_INDENT}}} catch (...) {{")
    lines.append(f"{_INDENT * 2}{probe.exit_function}(true{exit_args});")
    lines.append(f"{_INDENT * 2}throw;")
    lines.append(f"{_INDENT}}}")
    lines.append(f"{_INDENT}{probe.exit_function}(false{exit_args});")
    if not void:
        lines.append(f"{_INDENT}return {_RESULT_VAR};")
    lines.append("}")
    return lines


def _banner(config: Config) -> str:
    return (
        _COPYRIGHT
        + "\n// clang-format off\n"
        + f"// Generated by rocm_trace_gen {__version__} from {config.path.name}. Do not edit.\n"
        + "// Regenerate with shared/rocm-trace/scripts/rocm_trace_gen.py; see the README there.\n"
    )


def _header(config: Config, tables: Sequence[Table]) -> str:
    lines = [_banner(config), "#pragma once", ""]
    lines += [f"#include <{inc}>" for inc in config.includes]
    lines.append("")
    if config.capture_level:
        lines += [
            "// What the probes record: ROCM_TRACE_CAPTURE_NONE, _SCALARS or _FULL",
            "// (rocm/trace/values.h). Defined by the runtime and set from the",
            "// ROCM_TRACE_CAPTURE environment variable before the tables are installed.",
            f"extern int {config.capture_level};",
            "",
        ]
    for t in tables:
        lines += [
            "// Replace each non-null entry of *table, within its size, with its probe.",
            "// Not thread-safe: call before the table is published. Calling it again",
            "// is harmless. The original entries are kept in one copy per table type,",
            "// so install one instance of each table.",
            f"void rocm_trace_install_{t.name}({t.name}* table);",
            "",
        ]
    lines.append("// clang-format on")
    return "\n".join(lines) + "\n"


_HELPERS = [
    "// A string pointer that is never null, and its length within cap.",
    "inline const char* text(const char* s) {",
    f'{_INDENT}return s != nullptr ? s : "";',
    "}",
    "",
    "inline uint16_t text_size(const char* s, size_t cap) {",
    f"{_INDENT}return s != nullptr ? static_cast<uint16_t>(strnlen(s, cap)) : 0;",
    "}",
    "",
    "// N zero values, for array fields that are not captured.",
    "template <typename T, size_t N>",
    "const T* zeros() {",
    f"{_INDENT}static const T values[N] = {{}};",
    f"{_INDENT}return values;",
    "}",
    "",
    "constexpr bool has_slot(size_t limit, size_t offset) {",
    f"{_INDENT}return offset + sizeof(void*) <= limit;",
    "}",
    "",
]


def _source(
    config: Config,
    tables: Sequence[Table],
    probes: Dict[str, List[_Probe]],
    skipped: Dict[str, List[Tuple[Slot, str]]],
) -> str:
    provider = config.provider
    lines = [
        _banner(config),
        f'#include "{config.output_name}.h"',
        "",
        "#include <rocm/trace/trace.h>",
        "",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <string.h>",
        "",
        f"TRACELOGGING_DECLARE_PROVIDER({provider});",
        "",
    ]
    for t in tables:
        lines.append(
            f"static_assert(sizeof({t.name}) == offsetof({t.name}, {t.last_member}) + "
            f"sizeof({t.name}::{t.last_member}),"
        )
        lines.append(
            f'              "{t.name} changed after the trace probes were generated; '
            f'regenerate them");'
        )
    lines += ["", "namespace {", ""] + _HELPERS
    for t in tables:
        saved = f"saved_{t.name}"
        lines += [f"{t.name} {saved}{{}};", ""]
        for probe in probes[t.name]:
            lines += _probe_source(probe, provider, saved)
            lines.append("")
    lines += ["}  // namespace", ""]

    for t in tables:
        saved = f"saved_{t.name}"
        lines.append(f"void rocm_trace_install_{t.name}({t.name}* table) {{")
        lines.append(f"{_INDENT}if (table == nullptr) return;")
        lines.append(
            f"{_INDENT}auto const limit = static_cast<size_t>(table->{t.size_member});"
        )
        for probe in probes[t.name]:
            m = probe.slot.member
            lines.append(
                f"{_INDENT}if (has_slot(limit, offsetof({t.name}, {m})) && table->{m} != nullptr &&"
            )
            lines.append(f"{_INDENT * 2}table->{m} != {probe.function}) {{")
            lines.append(f"{_INDENT * 2}{saved}.{m} = table->{m};")
            lines.append(f"{_INDENT * 2}table->{m} = {probe.function};")
            lines.append(f"{_INDENT}}}")
        for slot, reason in skipped[t.name]:
            lines.append(f"{_INDENT}// {slot.member}: not traced ({reason})")
        lines.append("}")
        lines.append("")
    lines.append("// clang-format on")
    return "\n".join(lines) + "\n"


SCHEMA_VERSION = 1

# clang spells an unnamed struct, union or enum with its source location, which
# differs between machines: "(anonymous union at /path/x.h:64:3)".
_UNNAMED = re.compile(r"\((?:anonymous|unnamed) (struct|union|enum) at [^)]*\)")


def _type_name(spelling: str) -> str:
    return _UNNAMED.sub(r"(unnamed \1)", spelling)


def _field_schema(f: Field, enums: Dict[str, CType]) -> Dict[str, object]:
    """One field of the probe schema; enum types it uses are added to enums."""
    out: Dict[str, object] = {"name": f.name, "role": f.role, "param": f.param}
    if f.path:
        out["path"] = list(f.path)
    ctype = f.ctype
    out["type"] = _type_name(ctype.spelling if ctype is not None else f.param_type)
    if f.param_type and _type_name(f.param_type) != out["type"]:
        out["param_type"] = _type_name(f.param_type)
    encoding = f.macro[len("TraceLogging") :]
    out["encoding"] = encoding
    if encoding.endswith("FixedArray"):
        out["count"] = int(f.args[-1])
    if encoding == "CountedString":
        out["capped"] = True
    enum = ctype
    if enum is not None and enum.kind == Kind.ARRAY:
        enum = enum.element
    if enum is not None and enum.kind == Kind.ENUM:
        out["enum"] = _type_name(enum.spelling)
        enums.setdefault(_type_name(enum.spelling), enum)
    out["capture"] = f.capture
    if f.on_success:
        out["on_success"] = True
    return out


def _schema(
    config: Config,
    probes: Dict[str, List[_Probe]],
    skipped: Dict[str, List[Tuple[Slot, str]]],
) -> str:
    """The probe schema: every event's fields, with what a reader needs to turn
    them back into the call's arguments (C types, enumerator names, the
    members a struct was flattened into, what each capture level records).
    One field per line, so a changed API shows as a small diff."""
    enums: Dict[str, CType] = {}
    events: List[str] = []
    for probes_of_table in probes.values():
        for probe in probes_of_table:
            slot = probe.slot
            for phase, opcode, fields in (
                ("enter", "start", probe.enter),
                ("exit", "stop", probe.exit),
            ):
                head = {
                    "name": f"{slot.api}_{phase}",
                    "api": slot.api,
                    "table": slot.table,
                    "phase": phase,
                    "opcode": opcode,
                    "level": probe.level,
                    "keywords": list(probe.keywords),
                }
                rows = [json.dumps(_field_schema(f, enums)) for f in fields]
                body = ",\n".join(f"        {r}" for r in rows)
                events.append(
                    f"    {json.dumps(head)[:-1]},\n"
                    + (
                        f'      "fields": [\n{body}\n      ]}}'
                        if rows
                        else '      "fields": []}'
                    )
                )
    enum_rows = [
        f"    {json.dumps(name)}: "
        + json.dumps(
            {
                "signed": e.enum_signed,
                "size": e.size,
                "enumerators": {n: v for n, v in e.enumerators},
            }
        )
        for name, e in sorted(enums.items())
    ]
    skip_rows = [
        f"    {json.dumps({'api': slot.api, 'table': slot.table, 'reason': reason})}"
        for table_skips in skipped.values()
        for slot, reason in table_skips
    ]
    head = {
        "schema": "rocm-trace-probe-schema",
        "version": SCHEMA_VERSION,
        "generator": f"rocm_trace_gen {__version__}",
        "config": config.path.name,
        "provider": config.provider,
        "string_cap": STRING_CAP,
        "capture_levels": ["none", "scalars", "full"],
    }
    parts = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in head.items()]
    parts.append(
        '  "enums": {\n' + ",\n".join(enum_rows) + "\n  }"
        if enum_rows
        else '  "enums": {}'
    )
    parts.append(
        '  "skipped": [\n' + ",\n".join(skip_rows) + "\n  ]"
        if skip_rows
        else '  "skipped": []'
    )
    parts.append(
        '  "events": [\n' + ",\n".join(events) + "\n  ]" if events else '  "events": []'
    )
    return "{\n" + ",\n".join(parts) + "\n}\n"


def generate(config: Config, tables: Sequence[Table]) -> Dict[str, str]:
    """Return ``{file name: contents}`` for the header, the source and the
    probe schema (``<output_name>.json``).

    Raises GenerateError listing every problem found.
    """
    errors: List[str] = []
    apis_in_tables = {s.api for t in tables for s in t.slots}
    for name in sorted(set(config.apis) - apis_in_tables):
        errors.append(f"apis.{name}: no table has this entry")

    probes: Dict[str, List[_Probe]] = {t.name: [] for t in tables}
    skipped: Dict[str, List[Tuple[Slot, str]]] = {t.name: [] for t in tables}
    for t in tables:
        for slot in t.slots:
            api = config.apis.get(slot.api, ApiConfig())
            if api.skip:
                skipped[t.name].append((slot, api.reason))
                continue
            probe, problems = _plan(slot, api, config)
            errors.extend(problems)
            if probe is not None:
                probes[t.name].append(probe)

    if errors:
        raise GenerateError(errors)
    return {
        f"{config.output_name}.h": _header(config, tables),
        f"{config.output_name}.cpp": _source(config, tables, probes, skipped),
        f"{config.output_name}.json": _schema(config, probes, skipped),
    }
