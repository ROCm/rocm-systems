# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Persistent, cross-process cache for expensive test-setup probes.

This module stores the results of those probes in a single JSON file that all
the CTest-spawned processes share. The first process that needs a value pays for
it once and writes it; every later process reads it back.

Set ``ROCPROFSYS_DISABLE_TEST_CACHE=1`` to disable the cache entirely.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import functools
import getpass
import hashlib
import importlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

_DISABLE_ENV = "ROCPROFSYS_DISABLE_TEST_CACHE"

# Set only by disable_for_process() to turn off cache for remainder of this process
_proc_cache_disabled = False

# Debug aid, logs HIT/MISS lines for cache lookup attempts.
_DEBUG_ENV = "ROCPROFSYS_TEST_CACHE_DEBUG"

_logger = logging.getLogger("rocprofsys.syscache")


def _configure_debug_logger() -> None:
    """Enable ``[syscache] ...`` debug output when ``_DEBUG_ENV`` is truthy."""
    if os.environ.get(_DEBUG_ENV, "").strip() not in ("1", "ON", "on", "true", "True"):
        return
    _logger.setLevel(logging.DEBUG)
    if not _logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[syscache] %(message)s"))
        _logger.addHandler(handler)
        _logger.propagate = False


_configure_debug_logger()


def _debug_log(event: str, cache_key: str) -> None:
    """Log a ``[syscache] HIT/MISS: <key>`` line when debugging is enabled."""
    _logger.debug("%s: %s", event, cache_key)


# ---------------------------------------------------------------------------
# Codec errors
# ---------------------------------------------------------------------------
class CacheError(Exception):
    """Base class for errors raised by the cache codec."""


class SerializationError(CacheError):
    """A value cannot be encoded for the cache (generally a bug)."""


class DeserializationError(CacheError):
    """A cached payload cannot be decoded (corrupt/stale/incompatible entry)."""


# ---------------------------------------------------------------------------
# Typed JSON codec
#
# JSON natively supports null, booleans, numbers, strings, lists, and dicts.
# Wrap other types as tagged objects:
# {"__type__": "<kind>", "<value>": <payload>}
#
# Dataclasses are handled automatically: any dataclass instance is stored by
# value (its ``init=True`` fields, each recursively encoded) tagged with the
# defining ``module``/``qualname``, and decode() re-imports that class to rebuild
# it.
# ---------------------------------------------------------------------------
def encode(obj: Any) -> Any:
    # No lookup table: we dispatch by subclass, not exact type.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Path):
        return {"__type__": "path", "value": str(obj)}
    if isinstance(obj, tuple):
        return {"__type__": "tuple", "value": [encode(x) for x in obj]}
    if isinstance(obj, set):
        return {"__type__": "set", "value": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        return {
            "__type__": "dict",
            "value": [[encode(k), encode(v)] for k, v in obj.items()],
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            "__type__": "dataclass",
            "module": type(obj).__module__,
            "qualname": type(obj).__qualname__,
            "value": {
                f.name: encode(getattr(obj, f.name))
                for f in dataclasses.fields(obj)
                if f.init
            },
        }
    raise SerializationError(f"cannot cache value of type {type(obj).__name__}")


def _resolve_dataclass(module_name: str, qualname: str) -> type:
    """Import and return the dataclass named ``module_name.qualname``.

    Raises ``DeserializationError`` (which callers treat as a cache miss) if the
    class can't be imported or isn't a dataclass.
    """
    try:
        resolved: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            resolved = getattr(resolved, part)
    except (ImportError, AttributeError) as exc:
        raise DeserializationError(
            f"cannot resolve cached dataclass {module_name}.{qualname}"
        ) from exc
    if not (isinstance(resolved, type) and dataclasses.is_dataclass(resolved)):
        raise DeserializationError(f"{module_name}.{qualname} is not a dataclass")
    return resolved


def _decode_dataclass(obj: dict) -> Any:
    cls = _resolve_dataclass(obj.get("module"), obj.get("qualname"))
    return cls(**{name: decode(v) for name, v in obj["value"].items()})


_DECODERS: dict[str, Callable[[dict], Any]] = {
    "path": lambda obj: Path(obj["value"]),
    "tuple": lambda obj: tuple(decode(x) for x in obj["value"]),
    "set": lambda obj: set(decode(x) for x in obj["value"]),
    "dict": lambda obj: {decode(k): decode(v) for k, v in obj["value"]},
    "dataclass": _decode_dataclass,
}


def decode(obj: Any) -> Any:
    if isinstance(obj, list):
        return [decode(x) for x in obj]
    if isinstance(obj, dict):
        kind = obj.get("__type__")
        handler = _DECODERS.get(kind)
        if handler is None:
            raise DeserializationError(f"unknown cache tag: {kind!r}")
        return handler(obj)
    return obj


# ---------------------------------------------------------------------------
# PersistentCache
# ---------------------------------------------------------------------------
class PersistentCache:
    """A JSON key/value store shared across processes for one build tree."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Lock a separate sidecar file, not the cache file itself: _flush
        # swaps the cache file's inode via os.replace and flock locks the inode,
        # so locking the cache file would not serialize concurrent writers
        # (lost writes in CTest parallel mode).
        self.lock_path = path.with_name(f"{path.stem}.lock{path.suffix}")
        self._entries: dict[str, Any] = self._read_entries_from_disk()

    def get(self, key: str) -> tuple[bool, Any]:
        """Return ``(found, value)``. ``found`` is False on miss or any error.

        A decode failure is fail-open (treated as a miss)
        """
        if key not in self._entries:
            return False, None
        try:
            return True, decode(self._entries[key])
        except (DeserializationError, ValueError, TypeError, KeyError):
            return False, None

    def clear(self) -> None:
        """Drop all entries, in memory and on disk.

        Used by the config setup step to force a full regeneration so stale
        capability data is never carried over from a previous run.
        """
        self._entries = {}
        try:
            with self._exclusive_lock():
                self._flush({"entries": {}})
        except OSError:
            pass  # fail-open: in-memory clear still forces recompute this process

    def set(self, key: str, value: Any) -> None:
        """Persist ``value`` under ``key`` (locked read-modify-write).

        A serialization failure propagates.
        """
        encoded = encode(value)

        self._entries[key] = encoded
        try:
            self._persist(key, encoded)
        except OSError:
            pass

    @contextlib.contextmanager
    def _exclusive_lock(self):
        """Hold an exclusive lock on the stable sidecar file for a whole
        read-modify-write operation.
        """
        # 0o700: the cache files are 0o600, so keep the per-user dir owner-only
        # too (defense-in-depth on shared /tmp; no effect if it already exists).
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lock_fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def _read_entries_from_disk(self) -> dict[str, Any]:
        """Return the on-disk entries mapping (empty on missing/corrupt/bad shape).

        Does not lock; callers needing consistency hold ``_exclusive_lock``.
        """
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}  # missing/corrupt -> start fresh
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data["entries"]
        return {}

    def _flush(self, data: dict[str, Any]) -> None:
        """Replace the cache file with ``data`` atomically (caller holds the lock).

        It is placed in the target directory so ``os.replace`` stays on one
        filesystem (atomic swap), and named ``<cachefile>.*.tmp`` so the suite's
        cleanup still sweeps it if a crash leaves it behind.
        """
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, prefix=f"{self.path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except BaseException:
            # mkstemp makes a fresh file each call, so a leftover would accumulate;
            # remove it on any failure (os.replace consumes it on success).
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _persist(self, key: str, encoded: Any) -> None:
        """Merge ``key`` into the on-disk cache under an exclusive lock.

        Re-reads current entries before writing so a concurrent writer's
        entries are preserved (CTest parallel mode).
        """
        with self._exclusive_lock():
            entries = self._read_entries_from_disk()
            entries[key] = encoded
            self._flush({"entries": entries})


# ---------------------------------------------------------------------------
# Cache file location
#
# The config-setup CTest fixture clears and fully regenerates the cache at the
# start of every run, and the cleanup fixture deletes the file at the end, so
# freshness does not depend on a content signature.
# ---------------------------------------------------------------------------
def _cache_discriminator() -> str:
    """Short hash identifying the build/install tree this cache belongs to."""
    root = (
        os.environ.get("ROCPROFSYS_BUILD_DIR")
        or os.environ.get("ROCPROFSYS_INSTALL_DIR")
        # Fallback: the pytest package is copied into each build/install tree,
        # so its location distinguishes builds when neither env var is set.
        or str(Path(__file__).resolve().parent)
    )
    return hashlib.sha256(root.encode("utf-8")).hexdigest()[:16]


def resolve_username() -> str:
    """Best-effort current username for building per-user tmp paths.

    Falls back to the numeric uid, then ``"0"``.
    """
    try:
        return getpass.getuser()
    except Exception:
        try:
            return str(os.getuid())
        except Exception:
            return "0"


def _cache_file_path(discriminator: str) -> Path:
    # Resolve the tmp base exactly like the suite's cleanup step
    # (ROCPROFSYS_TMPDIR -> TMPDIR -> /tmp)
    base = os.environ.get("ROCPROFSYS_TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    try:
        uid = os.getuid()
    except AttributeError:  # pragma: no cover - non-POSIX
        uid = 0
    # Nest under the user's name (``/tmp/$USER`` by default)
    filename = f"rocprofsys-syscache-{uid}-{discriminator}.tmp"
    return Path(base) / resolve_username() / filename


def disable_for_process() -> None:
    """Disable the shared cache for the remainder of this process.

    Used by the CTest *generation* pass (``--ctest-mode generate``)
    """
    global _proc_cache_disabled
    _proc_cache_disabled = True
    get_shared_cache.cache_clear()


@functools.lru_cache(maxsize=1)
def get_shared_cache() -> Optional[PersistentCache]:
    """Return the process-wide cache, or ``None`` if disabled/unavailable.

    The result is memoized so the file is read at most once per process.
    """
    if _proc_cache_disabled:
        return None
    if os.environ.get(_DISABLE_ENV, "").strip() in ("1", "ON", "on", "true", "True"):
        return None
    try:
        return PersistentCache(_cache_file_path(_cache_discriminator()))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
def _args_key(args: tuple, kwargs: dict) -> str:
    """Stable string key for positional/keyword args (Paths -> str)."""

    def norm(value: Any) -> str:
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    parts = [norm(a) for a in args]
    parts += [f"{k}={norm(v)}" for k, v in sorted(kwargs.items())]
    return "(" + ",".join(parts) + ")"


def persistent_cache(
    key: str,
    *,
    to_cache: Optional[Callable[[Any], Any]] = None,
    from_cache: Optional[Callable[[Any], Any]] = None,
    method: bool = False,
) -> Callable:
    """Persistently memoize a function across processes.

    The cross-process analogue of ``functools.lru_cache``: a function decorator
    keyed on the call arguments. Unlike ``lru_cache`` it is unbounded (no
    eviction, like ``maxsize=None``) and keyed on the explicit ``key`` namespace
    plus the args rather than on the function object.

    ``to_cache``/``from_cache`` convert between the function's return type and a
    cache-serializable form

    The ``method`` flag selects how the cache key treats the first argument:

    - ``method=False`` (default) -- a free (module-level) function: every
      argument is part of the key.
    - ``method=True`` -- drops the first argument (``self`` for instance methods)

    For a no-argument property, use ``persistent_cached_property`` instead of
    this decorator.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            store = get_shared_cache()
            key_args = args[1:] if method else args
            cache_key = f"{key}:{_args_key(key_args, kwargs)}"
            if store is not None:
                try:
                    found, cached = store.get(cache_key)
                    if found:
                        _debug_log("HIT", cache_key)
                        return from_cache(cached) if from_cache else cached
                except Exception:
                    pass  # fall through to compute
                _debug_log("MISS", cache_key)

            result = func(*args, **kwargs)

            if store is not None:
                # Serialization errors propagate on purpose (see PersistentCache.set);
                # write/lock errors are already handled fail-open inside set().
                store.set(cache_key, to_cache(result) if to_cache else result)
            return result

        return wrapper

    return decorator


class persistent_cached_property:
    """``functools.cached_property`` that also persists across processes.

    When reading, it looks up the value in the following order:
     - instance ``__dict__`` (this process)
     - shared file cache (another process)
     - compute (if necessary, then written to shared file)
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.attrname: Optional[str] = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner, name: str) -> None:
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise TypeError(
                "Cannot assign the same persistent_cached_property to two names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError("persistent_cached_property used without __set_name__")

        try:
            inst_dict = instance.__dict__
        except AttributeError:
            return self.func(instance)

        if self.attrname in inst_dict:
            return inst_dict[self.attrname]

        # cap.<module>.<ClassQualName>.<attr>. Including __module__ (not just the
        # qualname) prevents two same-named classes in different modules from
        # colliding on one key when they share the cache file. Note the key is
        # deliberately independent of instance/env state: within a single CTest
        # session the config and environment are constant, and keying on them
        # would defeat the cross-process sharing this cache exists for.
        klass = type(instance)
        cache_key = f"cap.{klass.__module__}.{klass.__qualname__}.{self.attrname}"
        store = get_shared_cache()
        if store is not None:
            try:
                found, cached = store.get(cache_key)
                if found:
                    _debug_log("HIT", cache_key)
                    inst_dict[self.attrname] = cached
                    return cached
            except Exception:
                pass
            _debug_log("MISS", cache_key)

        value = self.func(instance)

        if store is not None:
            store.set(cache_key, value)
        inst_dict[self.attrname] = value
        return value
