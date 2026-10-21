# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Parse ML API selector flags for profile and analyze."""

from typing import Dict, List, NamedTuple, Optional, Tuple

PROFILE_SELECTORS: Tuple[str, ...] = ("torch", "triton", "jax")
INJECTABLE_SELECTORS: Tuple[str, ...] = ("torch", "triton")
ANALYZE_SELECTORS: Tuple[str, ...] = ("torch", "triton", "jax", "user")
DEFAULT_PROFILE_SELECTORS: Tuple[str, ...] = ("torch", "triton", "jax")
JAX_NOT_COLLECTED_WARNING = (
    "JAX tracing is not available yet and will not be collected."
)
_PROFILE_SELECTOR_VALUES = "Values: torch, triton, jax."
_ANALYZE_SELECTOR_VALUES = "Values: torch, triton, jax, user."


class ProfileSelection(NamedTuple):
    saved: List[str]
    injectable: List[str]
    marker_trace: bool
    warning: Optional[str]


class OperatorPatterns(NamedTuple):
    explicit: Dict[str, List[str]]
    unscoped: List[str]
    jax_requested: bool


def split_selector_tokens(raw_values: List[str]) -> List[str]:
    """Split argv tokens on commas and drop empty pieces."""
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


def resolve_profile_selection(raw: Optional[List[str]]) -> ProfileSelection:
    """Resolve --trace-ml-api into saved selectors, injection, and marker trace."""
    if raw is None:
        return ProfileSelection([], [], False, None)
    if not raw:
        saved = list(DEFAULT_PROFILE_SELECTORS)
        injectable = [name for name in saved if name in INJECTABLE_SELECTORS]
        return ProfileSelection(saved, injectable, True, None)

    tokens = split_selector_tokens(raw)
    if not tokens:
        saved = list(DEFAULT_PROFILE_SELECTORS)
        injectable = [name for name in saved if name in INJECTABLE_SELECTORS]
        return ProfileSelection(saved, injectable, True, None)

    for name in tokens:
        if name == "user":
            raise ValueError(
                "user is not a --trace-ml-api value. "
                "User-defined regions are included with every --trace-ml-api run."
            )
        if name not in PROFILE_SELECTORS:
            raise ValueError(
                f"Unknown --trace-ml-api value {name!r}. {_PROFILE_SELECTOR_VALUES}"
            )

    saved = _unique_keep_order(tokens)
    injectable = [name for name in saved if name in INJECTABLE_SELECTORS]
    warning = JAX_NOT_COLLECTED_WARNING if "jax" in saved else None
    return ProfileSelection(saved, injectable, True, warning)


def resolve_list_selectors(raw: Optional[List[str]]) -> Optional[List[str]]:
    """Resolve --list-operators. Empty means every collected selector."""
    if raw is None:
        return None
    tokens = split_selector_tokens(raw)
    for name in tokens:
        if name not in ANALYZE_SELECTORS:
            raise ValueError(
                f"Unknown --list-operators value {name!r}. {_ANALYZE_SELECTOR_VALUES}"
            )
    return _unique_keep_order(tokens)


def _split_operator_token(token: str) -> Tuple[Optional[str], List[str]]:
    if ":" in token:
        prefix, rest = token.split(":", 1)
        if prefix in ANALYZE_SELECTORS:
            return prefix, _comma_patterns(rest)
    return None, _comma_patterns(token)


def resolve_operator_patterns(raw: Optional[List[str]]) -> Optional[OperatorPatterns]:
    """Resolve --operator into explicit prefixes, unscoped globs, and jax."""
    if raw is None:
        return None
    if not raw:
        return OperatorPatterns({}, ["**"], False)

    explicit: Dict[str, List[str]] = {}
    unscoped: List[str] = []
    jax_requested = False
    for value in raw:
        token = str(value).strip()
        if not token:
            continue
        prefix, patterns = _split_operator_token(token)
        if prefix == "jax":
            jax_requested = True
            continue
        if not patterns:
            if prefix is None:
                continue
            patterns = ["**"]
        if prefix is None:
            unscoped.extend(patterns)
        else:
            explicit.setdefault(prefix, []).extend(patterns)
    if not explicit and not unscoped and not jax_requested:
        return OperatorPatterns({}, ["**"], False)
    return OperatorPatterns(explicit, unscoped, jax_requested)
