"""Provider registry — name → class mapping.

Providers self-register at import time via register(); callers resolve
names via get_provider(). list_providers() returns the set of available
names for CLI display (`perfxpert providers list`).

Threading: the registry is populated at import time only; get/list are
read-only. No locking needed for the expected single-threaded import
model; tests that mutate should monkeypatch _REGISTRY.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Type

from perfxpert.providers._base import Provider

# Module-level registry: name → (cls, description)
_REGISTRY: Dict[str, Tuple[Type[Provider], str]] = {}


def register(name: str, cls: Type[Provider], description: str = "") -> None:
    """Register a provider class under a string name.

    Args:
        name: Short identifier (e.g., "anthropic", "openai").
        cls: Provider subclass.
        description: Human-readable blurb for CLI listings.
    """
    if not name:
        raise ValueError("provider name must be non-empty")
    if not isinstance(cls, type) or not issubclass(cls, Provider):
        raise TypeError(f"{cls!r} is not a Provider subclass")
    _REGISTRY[name] = (cls, description)


def get_provider(name: str, **kwargs: Any) -> Provider:
    """Instantiate the provider registered under `name`.

    Args:
        name: Registered provider name.
        **kwargs: Forwarded to the provider constructor.

    Raises:
        KeyError: if no provider is registered under `name`.
    """
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"Unknown provider {name!r}; known: {known}")
    cls, _description = _REGISTRY[name]
    return cls(**kwargs)


def list_providers() -> Dict[str, str]:
    """Return a copy of the name→description map."""
    return {name: desc for name, (_cls, desc) in _REGISTRY.items()}


__all__ = ["register", "get_provider", "list_providers"]
