# Contributing: new tool

## What you're adding

A deterministic Python function that an agent can call. Tools are the
pure-function layer between agents and data — no LLM calls, no I/O except
`knowledge/*.yaml` loads and SQL reads.

## File locations

- Implementation: `perfxpert/tools/<module>.py`
- Tests: `tests/test_tools/test_<module>.py`

## Template

```python
"""<module> — <one-line purpose>.

Tool class: READ_ONLY (MCP-safe) or EXECUTION (in-process only).

Tool classes are enumerated in the multi-agent design spec (local
contributor doc, not tracked in the repo) — Appendix A.
"""

from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def my_tool(arg: str) -> Dict[str, Any]:
    """<docstring>

    Args:
        arg: <what>

    Returns:
        <shape>

    Raises:
        KeyError: if <condition>
    """
    data = load_yaml("my_yaml")
    # ...
    return {...}
```

## Schema constraints (CI-enforced)

- Type hints mandatory on args + return
- Docstring in Google style
- No external I/O (filesystem writes, network) without `ToolClass.EXECUTION`
- < 100 ms per call (p99) — covered by a timing test

## Tests you must add

- `test_<name>_returns_expected_shape` — happy path
- `test_<name>_raises_on_unknown_input` — error path
- `test_<name>_is_read_only_class` — MCP policy
- `test_<name>_under_50ms` — performance

## Review requirements

- 1 core reviewer
- CI green (unit + MCP-exposure + type check)

## Common pitfalls

- Don't re-implement what a `knowledge/*.yaml` already covers — look up instead
- `@tool_class` decorator MUST come after docstring (otherwise Pydantic schema generation skips it)
- Avoid f-string SQL for integer bound-params — use f-strings for INTEGER WHERE/LIMIT
- Pure Python stdlib + yaml + Pydantic only — no heavy deps

## Related docs

- Design spec: Appendix A (tool inventory)
- Existing tools under `perfxpert/tools/` as references
