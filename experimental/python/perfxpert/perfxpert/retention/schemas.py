"""Typed contracts for PerfXpert's retained-observation subsystem."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

PersistenceStatus = Literal["persisted", "disabled", "quota_exceeded", "error"]


@dataclass(frozen=True)
class ScopeContext:
    """Stable project namespace resolved once at a reader/writer boundary."""

    scope_id: str
    root: str
    label: str


@dataclass(frozen=True)
class SourceSnapshot:
    """Privacy-safe metadata fingerprint plus a transient local source path."""

    role: str
    ordinal: int
    resolved_path: str
    path_hash: str
    exists: bool
    is_directory: bool
    size: int
    mtime_ns: int
    sqlite_change_counter: Optional[int]
    sqlite_page_count: Optional[int]
    sqlite_schema_cookie: Optional[int]
    sqlite_version_valid_for: Optional[int]
    directory_entries_hash: Optional[str]
    wal_size: int
    wal_mtime_ns: int
    wal_header_hash: Optional[str]

    @classmethod
    def capture(
        cls,
        path: Union[str, Path],
        *,
        role: str,
        ordinal: int = 0,
    ) -> "SourceSnapshot":
        """Capture cheap source metadata without modifying the source."""

        raw = os.fspath(path) if path is not None else ""
        if raw:
            resolved = str(Path(raw).expanduser().resolve(strict=False))
        else:
            resolved = ""
        path_hash = hashlib.sha256(b"perfxpert-source-path-v1\0" + resolved.encode("utf-8")).hexdigest()

        exists = False
        is_directory = False
        size = 0
        mtime_ns = 0
        header_values: Tuple[Optional[int], Optional[int], Optional[int], Optional[int]] = (
            None,
            None,
            None,
            None,
        )
        directory_entries_hash: Optional[str] = None
        if resolved:
            try:
                stat = Path(resolved).stat()
                is_directory = Path(resolved).is_dir()
                is_file = Path(resolved).is_file()
                exists = is_file or is_directory
                size = int(stat.st_size)
                mtime_ns = int(stat.st_mtime_ns)
                if is_file:
                    header_values = _read_sqlite_header(Path(resolved))
                elif is_directory:
                    entries = sorted(child.name for child in Path(resolved).iterdir())
                    directory_entries_hash = hashlib.sha256("\0".join(entries).encode("utf-8")).hexdigest()
            except OSError:
                pass

        wal_size = 0
        wal_mtime_ns = 0
        wal_header_hash: Optional[str] = None
        if resolved and not is_directory:
            wal_path = Path(resolved + "-wal")
            try:
                wal_stat = wal_path.stat()
                if wal_path.is_file():
                    wal_size = int(wal_stat.st_size)
                    wal_mtime_ns = int(wal_stat.st_mtime_ns)
                    with wal_path.open("rb") as stream:
                        wal_header_hash = hashlib.sha256(stream.read(32)).hexdigest()
            except OSError:
                pass

        return cls(
            role=str(role),
            ordinal=int(ordinal),
            resolved_path=resolved,
            path_hash=path_hash,
            exists=exists,
            is_directory=is_directory,
            size=size,
            mtime_ns=mtime_ns,
            sqlite_change_counter=header_values[0],
            sqlite_page_count=header_values[1],
            sqlite_schema_cookie=header_values[2],
            sqlite_version_valid_for=header_values[3],
            directory_entries_hash=directory_entries_hash,
            wal_size=wal_size,
            wal_mtime_ns=wal_mtime_ns,
            wal_header_hash=wal_header_hash,
        )

    def identity_fields(self) -> Dict[str, Any]:
        """Return the fields that participate in retained-record identity."""

        return {
            "role": self.role,
            "ordinal": self.ordinal,
            "path_hash": self.path_hash,
            "exists": self.exists,
            "is_directory": self.is_directory,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "sqlite_change_counter": self.sqlite_change_counter,
            "sqlite_page_count": self.sqlite_page_count,
            "sqlite_schema_cookie": self.sqlite_schema_cookie,
            "sqlite_version_valid_for": self.sqlite_version_valid_for,
            "directory_entries_hash": self.directory_entries_hash,
            "wal_size": self.wal_size,
            "wal_mtime_ns": self.wal_mtime_ns,
            "wal_header_hash": self.wal_header_hash,
        }

    def stored_fields(
        self,
        *,
        scope: ScopeContext,
        store_paths: bool,
    ) -> Dict[str, Any]:
        """Return policy-compliant provenance suitable for ``payload_json``."""

        display_path = ""
        redacted = False
        if self.resolved_path:
            if store_paths:
                display_path = self.resolved_path
            else:
                source_path = Path(self.resolved_path)
                try:
                    display_path = str(source_path.relative_to(Path(scope.root)))
                except ValueError:
                    display_path = source_path.name
                redacted = display_path != self.resolved_path
        return {
            **self.identity_fields(),
            "display_path": display_path,
            "provenance_redacted": redacted,
        }

    def same_revision(self, other: "SourceSnapshot") -> bool:
        """Return whether a post-compute snapshot matches this snapshot."""

        return self.identity_fields() == other.identity_fields()


@dataclass(frozen=True)
class RetentionPolicy:
    """Resolved retention configuration with no initialization side effects."""

    enabled: bool
    root: str
    max_bytes: int
    store_paths: bool
    scope: ScopeContext
    busy_timeout_ms: int = 500

    @property
    def db_path(self) -> Path:
        return Path(self.root) / "knowledge.db"


@dataclass(frozen=True)
class PersistenceReceipt:
    """Observable result of an explicit persistence attempt."""

    status: PersistenceStatus
    record_id: Optional[str]
    scope_id: str
    external_id: Optional[str] = None
    provenance_redacted: bool = False
    detail: Optional[str] = None

    @property
    def persisted(self) -> bool:
        return self.status == "persisted"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "record_id": self.record_id,
            "scope_id": self.scope_id,
            "external_id": self.external_id,
            "provenance_redacted": self.provenance_redacted,
            "detail": self.detail,
        }


def _read_sqlite_header(
    path: Path,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    try:
        with path.open("rb") as stream:
            header = stream.read(100)
    except OSError:
        return (None, None, None, None)
    if len(header) < 100 or header[:16] != b"SQLite format 3\0":
        return (None, None, None, None)

    def _u32(offset: int) -> int:
        return int.from_bytes(header[offset : offset + 4], "big", signed=False)

    return (_u32(24), _u32(28), _u32(40), _u32(92))


__all__ = [
    "PersistenceReceipt",
    "PersistenceStatus",
    "RetentionPolicy",
    "ScopeContext",
    "SourceSnapshot",
]
