"""Project scope and retention-policy resolution without filesystem writes."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union

from perfxpert.config import PerfXpertConfig, load_config
from perfxpert.retention.identity import domain_hash
from perfxpert.retention.schemas import RetentionPolicy, ScopeContext

_PROJECT_MARKERS = (".git", ".hg", ".svn")


def resolve_scope_context(
    *,
    project_root: Optional[Union[str, Path]] = None,
    source_dir: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
) -> ScopeContext:
    """Resolve the same path-bound project namespace for readers and writers."""

    explicit = project_root or os.environ.get("PERFXPERT_PROJECT_ROOT")
    if explicit:
        root = _canonical_path(explicit)
    else:
        working = _canonical_path(cwd or Path.cwd())
        search_start = _canonical_path(source_dir) if source_dir else working
        root = _nearest_project_root(search_start) or _nearest_project_root(working) or working

    label = _sanitize_label(root.name or "project")
    scope_id = domain_hash("perfxpert-project-scope-v1", str(root))
    return ScopeContext(scope_id=scope_id, root=str(root), label=label)


def build_retention_policy(
    *,
    project_root: Optional[Union[str, Path]] = None,
    source_dir: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
    config: Optional[PerfXpertConfig] = None,
) -> RetentionPolicy:
    """Resolve config and paths without creating the store."""

    effective = config or load_config()
    configured_root = os.environ.get("PERFXPERT_KNOWLEDGE_ROOT")
    root = _canonical_path(configured_root) if configured_root else _canonical_path(Path.home() / ".perfxpert")
    scope = resolve_scope_context(
        project_root=project_root,
        source_dir=source_dir,
        cwd=cwd,
    )
    return RetentionPolicy(
        enabled=bool(effective.knowledge_retention),
        root=str(root),
        max_bytes=int(effective.knowledge_max_mb) * 1024 * 1024,
        store_paths=bool(effective.knowledge_store_paths),
        scope=scope,
    )


def _nearest_project_root(start: Path) -> Optional[Path]:
    current = start if start.is_dir() else start.parent
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate
    return None


def _canonical_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _sanitize_label(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized[:80] or "project"


__all__ = ["build_retention_policy", "resolve_scope_context"]
