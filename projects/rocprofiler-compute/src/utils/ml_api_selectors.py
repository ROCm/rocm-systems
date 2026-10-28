# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Parse --trace-ml-api, --list-operators, and --operator."""

from typing import Dict, List, Optional, Set, Tuple

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS

_OPERATOR_BACKENDS: Tuple[str, ...] = KNOWN_ML_API_BACKENDS + ("user",)
UNSCOPED_OPERATOR_KEY = "all_supported_backends"


def split_selector_tokens(raw_values: List[str]) -> List[str]:
    """Split on commas and drop empty pieces."""
    tokens: List[str] = []
    for value in raw_values:
        for piece in str(value).split(","):
            stripped = piece.strip()
            if stripped:
                tokens.append(stripped)
    return tokens


def _unique_keep_order(names: List[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique.append(name)
    return unique


def _comma_patterns(text: str) -> List[str]:
    return [piece.strip() for piece in text.split(",") if piece.strip()]


def resolve_profile_selection(raw: Optional[List[str]]) -> List[str]:
    """Resolve --trace-ml-api."""
    if raw is None:
        return []
    if not raw:
        return list(KNOWN_ML_API_BACKENDS)

    tokens = split_selector_tokens(raw)
    if not tokens:
        return list(KNOWN_ML_API_BACKENDS)

    for name in tokens:
        if name == "user":
            raise ValueError(
                "user is not a --trace-ml-api value. "
                "User-defined regions are included with every --trace-ml-api run."
            )
        if name not in KNOWN_ML_API_BACKENDS:
            raise ValueError(
                f"Unknown --trace-ml-api value {name!r}. "
                f"Values: {', '.join(KNOWN_ML_API_BACKENDS)}."
            )

    return _unique_keep_order(tokens)


def resolve_list_selectors(raw: Optional[List[str]]) -> Optional[List[str]]:
    """Resolve --list-operators."""
    if raw is None:
        return None
    tokens = split_selector_tokens(raw)
    for name in tokens:
        if name not in _OPERATOR_BACKENDS:
            raise ValueError(
                f"Unknown --list-operators value {name!r}. "
                f"Values: {', '.join(_OPERATOR_BACKENDS)}."
            )
    return _unique_keep_order(tokens)


def _patterns_by_backend(token: str) -> Dict[str, List[str]]:
    """Split one --operator token into backend glob patterns."""
    token = token.replace(" ", ",")
    selector_tokens = _comma_patterns(token)
    prefix: Optional[str] = None
    selections: Dict[str, List[str]] = {}

    def _save(backend: Optional[str], pattern: str) -> None:
        key = backend if backend else UNSCOPED_OPERATOR_KEY
        selections.setdefault(key, []).append(pattern)

    for selector_token in selector_tokens:
        if "::" in selector_token and selector_token.find("::") <= selector_token.find(
            ":"
        ):
            _save(prefix, selector_token)
            continue
        if ":" in selector_token:
            prefix, rest = selector_token.split(":", 1)
            if prefix not in _OPERATOR_BACKENDS:
                raise ValueError(
                    f"Unknown --operator prefix {prefix!r}. "
                    f"Values: {', '.join(_OPERATOR_BACKENDS)}."
                )
            _save(prefix, rest.strip() or "**")
            continue
        _save(prefix, selector_token)
    return selections


def resolve_operator_patterns(
    raw: Optional[List[str]],
) -> Optional[Dict[str, List[str]]]:
    """Resolve --operator into backend glob patterns."""
    if raw is None:
        return None
    token = " ".join(str(value).strip() for value in raw if str(value).strip())
    if not token:
        return {UNSCOPED_OPERATOR_KEY: ["**"]}
    selections = _patterns_by_backend(token)
    if not selections:
        return {UNSCOPED_OPERATOR_KEY: ["**"]}
    return selections


def check_list_operators_against_traced_frameworks(
    list_operators: List[str],
    traced_frameworks: List[str],
) -> None:
    """Reject --list-operators backends the workload did not collect."""
    _require_ml_api_trace(traced_frameworks)
    traced_framework_set = set(traced_frameworks)
    for backend in list_operators:
        _reject_untraced_named_backend(backend, traced_framework_set)


def operator_patterns_for_traced_frameworks(
    operator_patterns: Dict[str, List[str]],
    traced_frameworks: List[str],
) -> Dict[str, List[str]]:
    """Check --operator backends.

    Copy prefix-less patterns onto traced frameworks and user.
    """
    _require_ml_api_trace(traced_frameworks)
    traced_framework_set = set(traced_frameworks)
    for backend in operator_patterns:
        if backend == UNSCOPED_OPERATOR_KEY:
            continue
        _reject_untraced_named_backend(backend, traced_framework_set)

    patterns_by_backend: Dict[str, List[str]] = {}
    unscoped_patterns: List[str] = []
    for backend, patterns in operator_patterns.items():
        if backend == UNSCOPED_OPERATOR_KEY:
            unscoped_patterns.extend(patterns)
            continue
        patterns_by_backend.setdefault(backend, []).extend(patterns)
    if unscoped_patterns:
        for operator_backend in dict.fromkeys(traced_frameworks + ["user"]):
            patterns_by_backend.setdefault(operator_backend, []).extend(
                unscoped_patterns
            )
    return patterns_by_backend


def _require_ml_api_trace(traced_frameworks: List[str]) -> None:
    if not traced_frameworks:
        raise ValueError(
            "Workload was not profiled with --trace-ml-api. "
            "Cannot use --list-operators or --operator."
        )


def _reject_untraced_named_backend(
    backend: str, traced_framework_set: Set[str]
) -> None:
    if backend in KNOWN_ML_API_BACKENDS and backend not in traced_framework_set:
        raise ValueError(
            f'Workload was not profiled with "--trace-ml-api {backend}". '
            "Cannot use that selector with --list-operators or --operator."
        )
