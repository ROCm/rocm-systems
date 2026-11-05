# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Turn parsed tables and a configuration into the probe header and source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import __version__
from .config import RESULT, ApiConfig, Config
from .fields import Field, Unmappable, string, value
from .model import Kind, Slot, Table

_RESULT_VAR = "rocm_trace_result"
_COPYRIGHT = "// Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT\n"
_INDENT = "    "


class GenerateError(Exception):
    """The configuration does not fit the tables; one message per problem."""

    def __init__(self, errors: Sequence[str]):
        super().__init__("\n".join(errors))
        self.errors = list(errors)


@dataclass(frozen=True)
class _Probe:
    slot: Slot
    function: str
    aliases: Tuple[Tuple[str, str], ...]
    params: Tuple[Tuple[str, str], ...]  # (declared type, name)
    level: str
    keywords: Tuple[str, ...]
    enter: Tuple[Field, ...]
    exit: Tuple[Field, ...]


def _declarable(spelling: str) -> bool:
    """Whether ``<spelling> name`` is a valid declaration."""
    return "(" not in spelling and "[" not in spelling


def _plan(
    slot: Slot, api: ApiConfig, config: Config
) -> Tuple[Optional[_Probe], List[str]]:
    errors: List[str] = []
    where = f"{slot.table}.{slot.member} ({slot.api})"
    names = [p.name for p in slot.params]
    known = set(names) | {RESULT}

    for key in ("out", "strings", "omit"):
        for name in getattr(api, key):
            if name not in known or (key == "out" and name == RESULT):
                errors.append(
                    f"{where}: {key} names '{name}', which is not a parameter"
                )

    params = {p.name: p.ctype for p in slot.params}
    for name in api.out:
        ctype = params.get(name)
        if ctype is not None and (ctype.kind != Kind.POINTER or ctype.pointee is None):
            errors.append(
                f"{where}: out parameter '{name}' is not a pointer to a complete type"
            )
    for name in api.strings:
        ctype = slot.result if name == RESULT else params.get(name)
        if ctype is not None and not ctype.is_char_pointer:
            errors.append(
                f"{where}: strings names '{name}', which is not a char pointer"
            )
    if errors:
        return None, errors

    aliases: List[Tuple[str, str]] = []
    declared: List[Tuple[str, str]] = []
    for p in slot.params:
        spelling = p.ctype.spelling
        if not _declarable(spelling):
            alias = f"{slot.table}_{slot.api}_{p.name}_t"
            aliases.append((alias, spelling))
            spelling = alias
        declared.append((spelling, p.name))

    enter: List[Field] = []
    for p in slot.params:
        if p.name in api.omit:
            continue
        try:
            if p.name in api.strings and p.name not in api.out:
                enter.append(string(p.name, p.name))
            else:
                enter.extend(value(p.ctype, p.name, p.name))
        except Unmappable as err:
            errors.append(
                f"{where}: parameter '{p.name}' has no field mapping: {err}; "
                "add it to omit"
            )

    exit_fields: List[Field] = []
    if slot.result.kind != Kind.VOID and RESULT not in api.omit:
        try:
            if RESULT in api.strings:
                exit_fields.append(string(_RESULT_VAR, RESULT))
            else:
                exit_fields.extend(value(slot.result, _RESULT_VAR, RESULT))
        except Unmappable as err:
            errors.append(
                f"{where}: the result has no field mapping: {err}; add result to omit"
            )

    success = config.success.get(slot.result.spelling)
    for name in api.out:
        ctype = params[name]
        condition = f"{name} != nullptr"
        if success is not None:
            condition = f"{_RESULT_VAR} == {success} && {condition}"
        try:
            if name in api.strings:
                exit_fields.append(string(name, f"{name}_out").guarded(condition))
            else:
                exit_fields.extend(
                    f.guarded(condition)
                    for f in value(ctype.pointee, f"(*{name})", f"{name}_out")
                )
        except Unmappable as err:
            errors.append(
                f"{where}: out parameter '{name}' points to a type with no field mapping: {err}"
            )

    for event, event_fields in (("enter", enter), ("exit", exit_fields)):
        seen = set()
        for f in event_fields:
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
            function=f"probe_{slot.table}_{slot.api}",
            aliases=tuple(aliases),
            params=tuple(declared),
            level=api.level or config.level,
            keywords=("API", *api.keywords),
            enter=tuple(enter),
            exit=tuple(exit_fields),
        ),
        [],
    )


def _event(
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
    lines = [f"using {alias} = {spelling};" for alias, spelling in probe.aliases]
    params = ", ".join(f"{t} {n}" for t, n in probe.params)
    lines.append(f"{slot.result.spelling} {probe.function}({params}) {{")
    lines += _event(
        provider, f"{slot.api}_enter", probe, "WINEVENT_OPCODE_START", probe.enter
    )
    call = f"{saved}.{slot.member}({', '.join(n for _, n in probe.params)})"
    if slot.result.kind == Kind.VOID:
        lines.append(f"{_INDENT}{call};")
    else:
        lines.append(f"{_INDENT}auto const {_RESULT_VAR} = {call};")
    lines += _event(
        provider, f"{slot.api}_exit", probe, "WINEVENT_OPCODE_STOP", probe.exit
    )
    if slot.result.kind != Kind.VOID:
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
    for t in tables:
        lines.append(
            "// Replace each non-null entry of *table with its probe. Not thread-safe:"
        )
        lines.append(
            "// call before the table is published. Calling it again is harmless."
        )
        lines.append(f"void rocm_trace_install_{t.name}({t.name}* table);")
        lines.append("")
    lines.append("// clang-format on")
    return "\n".join(lines) + "\n"


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
    lines += [
        "",
        "namespace {",
        "",
        "constexpr bool has_slot(size_t limit, size_t offset) {",
        f"{_INDENT}return offset + sizeof(void*) <= limit;",
        "}",
        "",
    ]
    for t in tables:
        saved = f"saved_{t.name}"
        lines.append(f"{t.name} {saved}{{}};")
        lines.append("")
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


def generate(config: Config, tables: Sequence[Table]) -> Dict[str, str]:
    """Return ``{file name: contents}`` for the header and the source.

    Raises GenerateError listing every problem found.
    """
    errors: List[str] = []
    slots_by_api: Dict[str, List[Slot]] = {}
    for t in tables:
        for s in t.slots:
            slots_by_api.setdefault(s.api, []).append(s)

    for name in sorted(set(config.apis) - set(slots_by_api)):
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
    }
