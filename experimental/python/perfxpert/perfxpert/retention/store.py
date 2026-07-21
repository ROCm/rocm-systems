"""Versioned SQLite store for scoped, immutable performance observations."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from perfxpert.retention.identity import (
    canonical_json,
    dedupe_tags,
    payload_hash as compute_payload_hash,
    retained_record_id,
)
from perfxpert.retention.schemas import RetentionPolicy

APPLICATION_ID = 0x5058464B  # "PXFK"
STORE_SCHEMA_VERSION = 1
_ALLOWED_KINDS = ("prediction", "trace_analysis", "run_comparison")
_REQUIRED_TABLES = {
    "knowledge_scopes",
    "knowledge_observations",
    "knowledge_tags",
}
_EXPECTED_COLUMNS = {
    "knowledge_scopes": ("scope_id", "label", "created_at"),
    "knowledge_observations": (
        "record_id",
        "observation_key",
        "payload_hash",
        "kind",
        "schema_version",
        "producer_version",
        "scope_id",
        "external_id",
        "deterministic",
        "confidence",
        "created_at",
        "last_seen_at",
        "seen_count",
        "payload_json",
    ),
    "knowledge_tags": ("record_id", "tag_key", "tag_value"),
}
_EXPECTED_INDEXES = {
    "idx_knowledge_scope_kind_seen",
    "idx_knowledge_external",
    "idx_knowledge_observation_key",
    "idx_knowledge_tags_lookup",
}


class RetentionError(RuntimeError):
    """Base class for retained-observation failures."""


class RetentionDisabledError(RetentionError):
    pass


class StoreUnavailableError(KeyError, RetentionError):
    pass


class StoreCorruptError(RetentionError):
    pass


class WrongStoreError(RetentionError):
    pass


class UnsupportedSchemaError(RetentionError):
    pass


class ObservationNotFoundError(KeyError, RetentionError):
    pass


class ScopeMismatchError(KeyError, RetentionError):
    pass


class IdentityCollisionError(RetentionError):
    pass


class QuotaExceededError(RetentionError):
    pass


class ObservationStore:
    """Short-lived-connection SQLite observation store."""

    def __init__(self, policy: RetentionPolicy):
        self.policy = policy

    @property
    def db_path(self) -> Path:
        return self.policy.db_path

    def write_observation(
        self,
        *,
        record_id: str,
        observation_key: str,
        payload_hash: str,
        kind: str,
        producer_version: str,
        external_id: Optional[str],
        deterministic: bool,
        confidence: Optional[float],
        payload: Mapping[str, Any],
        tags: Iterable[tuple[str, str]] = (),
        schema_version: int = 1,
    ) -> Dict[str, Any]:
        if not self.policy.enabled:
            raise RetentionDisabledError("knowledge retention is disabled")
        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"unsupported observation kind: {kind!r}")

        payload_json = canonical_json(payload)
        normalized_tags = dedupe_tags(tags)
        now = _now_iso()

        with self._writer_connection() as conn:
            try:
                expected_payload_hash = compute_payload_hash(
                    {
                        "payload": payload,
                        "tags": normalized_tags,
                    }
                )
                if payload_hash != expected_payload_hash:
                    raise IdentityCollisionError("payload_hash does not match payload and tags")
                expected_record_id = retained_record_id(
                    scope_id=self.policy.scope.scope_id,
                    observation_key_value=observation_key,
                    payload_hash_value=payload_hash,
                )
                if record_id != expected_record_id:
                    raise IdentityCollisionError("record_id does not match scope, observation key, and payload hash")
                conn.execute("BEGIN IMMEDIATE")
                existing = conn.execute(
                    """
                    SELECT observation_key, payload_hash, kind, schema_version,
                           producer_version, scope_id, external_id, deterministic,
                           confidence, payload_json
                    FROM knowledge_observations
                    WHERE record_id = ?
                    """,
                    (record_id,),
                ).fetchone()
                if existing is not None:
                    expected = (
                        observation_key,
                        payload_hash,
                        kind,
                        int(schema_version),
                        producer_version,
                        self.policy.scope.scope_id,
                        external_id,
                        int(bool(deterministic)),
                        confidence,
                        payload_json,
                    )
                    actual = tuple(existing)
                    if actual != expected:
                        raise IdentityCollisionError(f"record_id {record_id!r} maps to different immutable content")
                    existing_tags = tuple(
                        (row["tag_key"], row["tag_value"])
                        for row in conn.execute(
                            """
                            SELECT tag_key, tag_value FROM knowledge_tags
                            WHERE record_id = ?
                            ORDER BY tag_key, tag_value
                            """,
                            (record_id,),
                        ).fetchall()
                    )
                    if existing_tags != normalized_tags:
                        raise IdentityCollisionError(f"record_id {record_id!r} maps to different immutable tags")
                    conn.execute(
                        """
                        UPDATE knowledge_observations
                        SET last_seen_at = ?, seen_count = seen_count + 1
                        WHERE record_id = ?
                        """,
                        (now, record_id),
                    )
                else:
                    self._check_quota(len(payload_json.encode("utf-8")))
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO knowledge_scopes
                            (scope_id, label, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            self.policy.scope.scope_id,
                            self.policy.scope.label,
                            now,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO knowledge_observations (
                            record_id, observation_key, payload_hash, kind,
                            schema_version, producer_version, scope_id, external_id,
                            deterministic, confidence, created_at, last_seen_at,
                            seen_count, payload_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                        """,
                        (
                            record_id,
                            observation_key,
                            payload_hash,
                            kind,
                            int(schema_version),
                            producer_version,
                            self.policy.scope.scope_id,
                            external_id,
                            int(bool(deterministic)),
                            confidence,
                            now,
                            now,
                            payload_json,
                        ),
                    )
                    conn.executemany(
                        """
                        INSERT INTO knowledge_tags (record_id, tag_key, tag_value)
                        VALUES (?, ?, ?)
                        """,
                        ((record_id, key, value) for key, value in normalized_tags),
                    )
                self._check_page_quota(conn)
                conn.commit()
            except sqlite3.OperationalError as exc:
                conn.rollback()
                if "full" in str(exc).lower():
                    raise QuotaExceededError("knowledge store reached its SQLite page ceiling") from exc
                raise
            except Exception:
                conn.rollback()
                raise

        return {"record_id": record_id}

    def get_observation(
        self,
        record_id: str,
        *,
        scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_scope = scope_id or self.policy.scope.scope_id
        with self._reader_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM knowledge_observations
                WHERE record_id = ? AND scope_id = ?
                """,
                (record_id, selected_scope),
            ).fetchone()
            if row is None:
                raise ObservationNotFoundError(f"knowledge observation {record_id!r} not found in current scope")
            return _row_to_observation(conn, row)

    def get_latest_by_external_id(
        self,
        *,
        kind: str,
        external_id: str,
        scope_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        selected_scope = scope_id or self.policy.scope.scope_id
        with self._reader_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM knowledge_observations
                WHERE scope_id = ? AND kind = ? AND external_id = ?
                ORDER BY last_seen_at DESC, record_id DESC
                LIMIT 1
                """,
                (selected_scope, kind, external_id),
            ).fetchone()
            if row is None:
                raise ObservationNotFoundError(f"{kind} {external_id!r} not found in current scope")
            return _row_to_observation(conn, row)

    def external_id_scopes(
        self,
        *,
        kind: str,
        external_id: str,
    ) -> List[str]:
        """Return matching scope IDs for internal error classification."""

        with self._reader_connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT scope_id FROM knowledge_observations
                WHERE kind = ? AND external_id = ?
                ORDER BY scope_id
                """,
                (kind, external_id),
            ).fetchall()
            return [str(row["scope_id"]) for row in rows]

    def query(
        self,
        *,
        kind: Optional[str] = None,
        kernel_name: Optional[str] = None,
        gfx_id: Optional[str] = None,
        change_type: Optional[str] = None,
        verdict: Optional[str] = None,
        limit: int = 50,
        scope_id: Optional[str] = None,
        all_scopes: bool = False,
    ) -> List[Dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 500))
        clauses: List[str] = []
        params: List[Any] = []
        if not all_scopes:
            clauses.append("o.scope_id = ?")
            params.append(scope_id or self.policy.scope.scope_id)
        if kind is not None:
            clauses.append("o.kind = ?")
            params.append(kind)

        tag_filters = (
            ("kernel", kernel_name),
            ("gfx_id", gfx_id),
            ("change_type", change_type),
            ("verdict", verdict),
        )
        for tag_key, value in tag_filters:
            if value is None:
                continue
            clauses.append("""
                EXISTS (
                    SELECT 1 FROM knowledge_tags t
                    WHERE t.record_id = o.record_id
                      AND t.tag_key = ?
                      AND t.tag_value = ?
                )
                """)
            params.extend((tag_key, str(value)))

        where = " AND ".join(clauses) if clauses else "1 = 1"
        sql = f"""
            SELECT o.* FROM knowledge_observations o
            WHERE {where}
            ORDER BY o.last_seen_at DESC, o.record_id DESC
            LIMIT ?
        """
        params.append(bounded_limit)

        with self._reader_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [_row_to_observation(conn, row) for row in rows]

    def stats(
        self,
        *,
        scope_id: Optional[str] = None,
        all_scopes: bool = False,
    ) -> Dict[str, Any]:
        clauses: List[str] = []
        params: List[Any] = []
        if not all_scopes:
            clauses.append("scope_id = ?")
            params.append(scope_id or self.policy.scope.scope_id)
        where = " AND ".join(clauses) if clauses else "1 = 1"

        with self._reader_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT kind, COUNT(*) AS records, COALESCE(SUM(seen_count), 0) AS observations
                FROM knowledge_observations
                WHERE {where}
                GROUP BY kind
                ORDER BY kind
                """,
                params,
            ).fetchall()
            total = conn.execute(
                f"""
                SELECT COUNT(*) AS records, COALESCE(SUM(seen_count), 0) AS observations
                FROM knowledge_observations
                WHERE {where}
                """,
                params,
            ).fetchone()
        return {
            "scope_id": None if all_scopes else (scope_id or self.policy.scope.scope_id),
            "all_scopes": bool(all_scopes),
            "records": int(total["records"]),
            "observations": int(total["observations"]),
            "by_kind": {
                row["kind"]: {
                    "records": int(row["records"]),
                    "observations": int(row["observations"]),
                }
                for row in rows
            },
            "database_bytes": self.database_size(),
        }

    def clear(
        self,
        *,
        scope_id: Optional[str] = None,
        all_scopes: bool = False,
        compact: bool = False,
    ) -> int:
        if not self.db_path.exists():
            return 0

        with self._writer_connection(enforce_quota=False) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                if all_scopes:
                    count = int(conn.execute("SELECT COUNT(*) FROM knowledge_observations").fetchone()[0])
                    conn.execute("DELETE FROM knowledge_scopes")
                else:
                    selected_scope = scope_id or self.policy.scope.scope_id
                    count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM knowledge_observations WHERE scope_id = ?",
                            (selected_scope,),
                        ).fetchone()[0]
                    )
                    conn.execute(
                        "DELETE FROM knowledge_scopes WHERE scope_id = ?",
                        (selected_scope,),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if compact:
                conn.execute("VACUUM")
        return count

    def prune(
        self,
        *,
        older_than: datetime,
        scope_id: Optional[str] = None,
        all_scopes: bool = False,
        compact: bool = False,
    ) -> int:
        if not self.db_path.exists():
            return 0

        cutoff = older_than.astimezone(timezone.utc).isoformat()
        with self._writer_connection(enforce_quota=False) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE")
                if all_scopes:
                    cursor = conn.execute(
                        "DELETE FROM knowledge_observations WHERE last_seen_at < ?",
                        (cutoff,),
                    )
                else:
                    cursor = conn.execute(
                        """
                        DELETE FROM knowledge_observations
                        WHERE scope_id = ? AND last_seen_at < ?
                        """,
                        (scope_id or self.policy.scope.scope_id, cutoff),
                    )
                count = int(cursor.rowcount if cursor.rowcount >= 0 else 0)
                conn.execute("""
                    DELETE FROM knowledge_scopes
                    WHERE NOT EXISTS (
                        SELECT 1 FROM knowledge_observations o
                        WHERE o.scope_id = knowledge_scopes.scope_id
                    )
                    """)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            if compact:
                conn.execute("VACUUM")
        return count

    def database_size(self) -> int:
        try:
            return int(self.db_path.stat().st_size)
        except OSError:
            return 0

    @contextmanager
    def _writer_connection(
        self,
        *,
        enforce_quota: bool = True,
    ) -> Iterator[sqlite3.Connection]:
        self._prepare_store_path()
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.policy.busy_timeout_ms / 1000.0,
                isolation_level=None,
            )
        except sqlite3.DatabaseError as exc:
            raise StoreCorruptError(f"cannot open retained-knowledge store: {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(self.policy.busy_timeout_ms)}")
            conn.execute("PRAGMA foreign_keys = ON")
            self._prevalidate_writer_target(conn)
            self._ensure_schema(conn)
            mode = str(conn.execute("PRAGMA journal_mode = DELETE").fetchone()[0]).lower()
            if mode != "delete":
                raise RetentionError(f"SQLite refused rollback journal mode: {mode}")
            conn.execute("PRAGMA synchronous = FULL")
            if enforce_quota:
                self._configure_quota(conn)
            self._enforce_store_permissions()
            yield conn
        except sqlite3.DatabaseError as exc:
            raise StoreCorruptError(f"retained-knowledge store failure: {exc}") from exc
        finally:
            conn.close()

    @contextmanager
    def _reader_connection(self) -> Iterator[sqlite3.Connection]:
        if not self.db_path.exists():
            raise StoreUnavailableError("retained-knowledge store does not exist")
        _require_secure_root(Path(self.policy.root))
        _require_real_owned_path(self.db_path, directory=False)
        uri = f"{self.db_path.resolve(strict=False).as_uri()}?mode=ro"
        try:
            conn = sqlite3.connect(
                uri,
                uri=True,
                timeout=self.policy.busy_timeout_ms / 1000.0,
            )
        except sqlite3.DatabaseError as exc:
            raise StoreCorruptError(f"cannot read retained-knowledge store: {exc}") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(self.policy.busy_timeout_ms)}")
            conn.execute("PRAGMA query_only = ON")
            conn.execute("PRAGMA foreign_keys = ON")
            self._validate_schema(conn)
            yield conn
        except sqlite3.DatabaseError as exc:
            raise StoreCorruptError(f"retained-knowledge store failure: {exc}") from exc
        finally:
            conn.close()

    def _prepare_store_path(self) -> None:
        root = Path(self.policy.root)
        root_preexisting = root.exists()
        try:
            root.mkdir(parents=True, mode=0o700, exist_ok=True)
        except FileExistsError:
            pass
        _require_real_owned_path(root, directory=True)
        if root_preexisting:
            _require_secure_root(root)
        elif os.name == "posix":
            os.chmod(root, 0o700)

        db_path = self.db_path
        if not db_path.exists():
            flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                fd = os.open(db_path, flags, 0o600)
            except FileExistsError:
                pass
            else:
                os.close(fd)
        _require_real_owned_path(db_path, directory=False)

    def _enforce_store_permissions(self) -> None:
        if os.name == "posix":
            os.chmod(Path(self.policy.root), 0o700)
            os.chmod(self.db_path, 0o600)

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute("BEGIN IMMEDIATE")
            application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            objects = _user_schema_objects(conn)

            if application_id not in (0, APPLICATION_ID):
                raise WrongStoreError(
                    f"configured database has application_id {application_id}, " f"expected {APPLICATION_ID}"
                )
            if user_version == 0 and objects:
                raise WrongStoreError("version-zero knowledge database contains schema objects")
            if user_version > STORE_SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"knowledge schema {user_version} is newer than supported " f"{STORE_SCHEMA_VERSION}"
                )

            if user_version == 0:
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                conn.execute(f"PRAGMA application_id = {APPLICATION_ID}")
                conn.execute(f"PRAGMA user_version = {STORE_SCHEMA_VERSION}")
                self._validate_schema(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        self._validate_schema(conn)

    def _prevalidate_writer_target(self, conn: sqlite3.Connection) -> None:
        """Reject unrelated/newer stores before beginning a write transaction."""

        try:
            conn.execute("BEGIN")
            application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            objects = _user_schema_objects(conn)
            if application_id not in (0, APPLICATION_ID):
                raise WrongStoreError(
                    f"configured database has application_id {application_id}, " f"expected {APPLICATION_ID}"
                )
            if user_version == 0 and objects:
                raise WrongStoreError("version-zero knowledge database contains schema objects")
            if user_version > STORE_SCHEMA_VERSION:
                raise UnsupportedSchemaError(
                    f"knowledge schema {user_version} is newer than supported " f"{STORE_SCHEMA_VERSION}"
                )
            if application_id == APPLICATION_ID and user_version == STORE_SCHEMA_VERSION:
                self._validate_schema(conn)
        finally:
            conn.rollback()

    def _validate_schema(self, conn: sqlite3.Connection) -> None:
        application_id = int(conn.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if application_id != APPLICATION_ID:
            raise WrongStoreError(f"configured database has application_id {application_id}, expected {APPLICATION_ID}")
        if user_version > STORE_SCHEMA_VERSION:
            raise UnsupportedSchemaError(
                f"knowledge schema {user_version} is newer than supported {STORE_SCHEMA_VERSION}"
            )
        if user_version != STORE_SCHEMA_VERSION:
            raise UnsupportedSchemaError(f"knowledge schema {user_version} requires a writer migration")
        tables = _user_tables(conn)
        if tables != _REQUIRED_TABLES:
            raise StoreCorruptError(
                "knowledge schema table mismatch: " f"expected {sorted(_REQUIRED_TABLES)}, got {sorted(tables)}"
            )
        for table, expected_columns in _EXPECTED_COLUMNS.items():
            actual_columns = tuple(row["name"] for row in conn.execute(f"PRAGMA table_info({table})"))
            if actual_columns != expected_columns:
                raise StoreCorruptError(f"knowledge schema columns for {table} do not match v1")
        indexes = _user_indexes(conn)
        if indexes != _EXPECTED_INDEXES:
            raise StoreCorruptError(
                "knowledge schema index mismatch: " f"expected {sorted(_EXPECTED_INDEXES)}, got {sorted(indexes)}"
            )
        for object_name, expected_sql in _EXPECTED_SCHEMA_SQL.items():
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?",
                (object_name,),
            ).fetchone()
            if row is None or _normalize_sql(row["sql"]) != _normalize_sql(expected_sql):
                raise StoreCorruptError(f"knowledge schema definition for {object_name} does not match v1")
        unexpected_objects = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type IN ('view', 'trigger') AND name NOT LIKE 'sqlite_%'
            """).fetchall()
        if unexpected_objects:
            raise StoreCorruptError("knowledge schema contains unexpected views or triggers")
        _validate_foreign_key(
            conn,
            table="knowledge_observations",
            from_column="scope_id",
            target_table="knowledge_scopes",
        )
        _validate_foreign_key(
            conn,
            table="knowledge_tags",
            from_column="record_id",
            target_table="knowledge_observations",
        )

    def _configure_quota(self, conn: sqlite3.Connection) -> None:
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        if page_count * page_size > self.policy.max_bytes:
            raise QuotaExceededError("knowledge store already exceeds the configured size ceiling")
        max_pages = max(1, self.policy.max_bytes // page_size)
        conn.execute(f"PRAGMA max_page_count = {int(max_pages)}")

    def _check_page_quota(self, conn: sqlite3.Connection) -> None:
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
        page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
        if page_count * page_size > self.policy.max_bytes:
            raise QuotaExceededError("knowledge store write exceeded the configured size ceiling")

    def _check_quota(self, incoming_payload_bytes: int) -> None:
        projected = self.database_size() + int(incoming_payload_bytes) + 4096
        if projected > self.policy.max_bytes:
            raise QuotaExceededError(f"knowledge store quota exceeded ({projected} > {self.policy.max_bytes} bytes)")


def _row_to_observation(conn: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise StoreCorruptError(f"observation {row['record_id']!r} has malformed payload JSON") from exc
    tags = conn.execute(
        """
        SELECT tag_key, tag_value FROM knowledge_tags
        WHERE record_id = ?
        ORDER BY tag_key, tag_value
        """,
        (row["record_id"],),
    ).fetchall()
    grouped_tags: Dict[str, List[str]] = {}
    for tag in tags:
        grouped_tags.setdefault(tag["tag_key"], []).append(tag["tag_value"])
    tag_pairs = tuple((tag["tag_key"], tag["tag_value"]) for tag in tags)
    expected_payload_hash = compute_payload_hash(
        {
            "payload": payload,
            "tags": tag_pairs,
        }
    )
    if expected_payload_hash != row["payload_hash"]:
        raise StoreCorruptError(f"observation {row['record_id']!r} failed payload integrity validation")
    expected_record_id = retained_record_id(
        scope_id=row["scope_id"],
        observation_key_value=row["observation_key"],
        payload_hash_value=row["payload_hash"],
    )
    if expected_record_id != row["record_id"]:
        raise StoreCorruptError(f"observation {row['record_id']!r} failed record identity validation")
    return {
        "record_id": row["record_id"],
        "observation_key": row["observation_key"],
        "payload_hash": row["payload_hash"],
        "kind": row["kind"],
        "schema_version": int(row["schema_version"]),
        "producer_version": row["producer_version"],
        "scope_id": row["scope_id"],
        "external_id": row["external_id"],
        "deterministic": bool(row["deterministic"]),
        "confidence": row["confidence"],
        "created_at": row["created_at"],
        "last_seen_at": row["last_seen_at"],
        "seen_count": int(row["seen_count"]),
        "payload": payload,
        "tags": grouped_tags,
    }


def _user_tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
    return {str(row[0]) for row in rows}


def _user_schema_objects(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    rows = conn.execute("""
        SELECT type, name FROM sqlite_master
        WHERE name NOT LIKE 'sqlite_%'
        """).fetchall()
    return {(str(row[0]), str(row[1])) for row in rows}


def _user_indexes(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("""
        SELECT name FROM sqlite_master
        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
        """).fetchall()
    return {str(row[0]) for row in rows}


def _validate_foreign_key(
    conn: sqlite3.Connection,
    *,
    table: str,
    from_column: str,
    target_table: str,
) -> None:
    rows = conn.execute(f"PRAGMA foreign_key_list({table})").fetchall()
    if not any(
        row["from"] == from_column and row["table"] == target_table and str(row["on_delete"]).upper() == "CASCADE"
        for row in rows
    ):
        raise StoreCorruptError(f"knowledge schema foreign key for {table}.{from_column} is invalid")


def _require_real_owned_path(path: Path, *, directory: bool) -> None:
    metadata = path.lstat()
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(metadata.st_mode):
        kind = "directory" if directory else "regular file"
        raise WrongStoreError(f"retained-knowledge path is not a real {kind}: {path}")
    if os.name == "posix" and hasattr(os, "getuid") and metadata.st_uid != os.getuid():
        raise WrongStoreError(f"refusing retained-knowledge path owned by another user: {path}")


def _require_secure_root(path: Path) -> None:
    _require_real_owned_path(path, directory=True)
    if os.name == "posix" and path.lstat().st_mode & 0o022:
        raise WrongStoreError(f"knowledge root must not be writable by group/other: {path}")


def _normalize_sql(value: str) -> str:
    return " ".join(str(value).split()).strip().lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA_STATEMENTS: Sequence[str] = (
    """
    CREATE TABLE knowledge_scopes (
        scope_id TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE knowledge_observations (
        record_id TEXT PRIMARY KEY,
        observation_key TEXT NOT NULL,
        payload_hash TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (
            kind IN ('prediction', 'trace_analysis', 'run_comparison')
        ),
        schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
        producer_version TEXT NOT NULL,
        scope_id TEXT NOT NULL REFERENCES knowledge_scopes(scope_id) ON DELETE CASCADE,
        external_id TEXT,
        deterministic INTEGER NOT NULL CHECK (deterministic IN (0, 1)),
        confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
        created_at TEXT NOT NULL,
        last_seen_at TEXT NOT NULL,
        seen_count INTEGER NOT NULL CHECK (seen_count >= 1),
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE knowledge_tags (
        record_id TEXT NOT NULL
            REFERENCES knowledge_observations(record_id) ON DELETE CASCADE,
        tag_key TEXT NOT NULL,
        tag_value TEXT NOT NULL,
        PRIMARY KEY (record_id, tag_key, tag_value)
    )
    """,
    """
    CREATE INDEX idx_knowledge_scope_kind_seen
    ON knowledge_observations(scope_id, kind, last_seen_at DESC, record_id DESC)
    """,
    """
    CREATE INDEX idx_knowledge_external
    ON knowledge_observations(scope_id, kind, external_id)
    """,
    """
    CREATE INDEX idx_knowledge_observation_key
    ON knowledge_observations(scope_id, observation_key)
    """,
    """
    CREATE INDEX idx_knowledge_tags_lookup
    ON knowledge_tags(tag_key, tag_value, record_id)
    """,
)

_EXPECTED_SCHEMA_SQL = {
    "knowledge_scopes": _SCHEMA_STATEMENTS[0],
    "knowledge_observations": _SCHEMA_STATEMENTS[1],
    "knowledge_tags": _SCHEMA_STATEMENTS[2],
    "idx_knowledge_scope_kind_seen": _SCHEMA_STATEMENTS[3],
    "idx_knowledge_external": _SCHEMA_STATEMENTS[4],
    "idx_knowledge_observation_key": _SCHEMA_STATEMENTS[5],
    "idx_knowledge_tags_lookup": _SCHEMA_STATEMENTS[6],
}


__all__ = [
    "APPLICATION_ID",
    "IdentityCollisionError",
    "ObservationNotFoundError",
    "ObservationStore",
    "QuotaExceededError",
    "RetentionDisabledError",
    "RetentionError",
    "ScopeMismatchError",
    "STORE_SCHEMA_VERSION",
    "StoreCorruptError",
    "StoreUnavailableError",
    "UnsupportedSchemaError",
    "WrongStoreError",
]
