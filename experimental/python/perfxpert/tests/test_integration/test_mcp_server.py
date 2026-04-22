"""Integration: MCP server constructs cleanly."""

import asyncio
import json
from pathlib import Path

import pytest


def test_server_module_imports():
    """Module must import whether or not MCP SDK is installed."""
    from mcp_server import server  # noqa: F401


def test_server_constructs_when_sdk_present():
    """Server constructs successfully when MCP SDK is available."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        pytest.skip("MCP SDK not installed")
    from mcp_server.server import build_server
    s = build_server()
    assert s is not None


def test_build_server_raises_without_sdk():
    """build_server() raises clean error if MCP SDK missing."""
    # This test verifies the adapter pattern works
    # (normally skipped since MCP is installed)
    try:
        import mcp  # noqa: F401
        pytest.skip("MCP SDK is installed; test only relevant when SDK missing")
    except ImportError:
        from mcp_server.server import build_server
        with pytest.raises(RuntimeError, match="MCP SDK not installed"):
            build_server()


class _FakeTextContent:
    def __init__(self, *, type: str, text: str):
        self.type = type
        self.text = text


class _FakeServer:
    def __init__(self, name: str):
        self.name = name
        self.list_handler = None
        self.call_handler = None

    def list_tools(self):
        def decorator(fn):
            self.list_handler = fn
            return fn

        return decorator

    def call_tool(self):
        def decorator(fn):
            self.call_handler = fn
            return fn

        return decorator


def _build_fake_server(monkeypatch, tools):
    from mcp_server import server as server_mod

    monkeypatch.setattr(server_mod, "_MCP_AVAILABLE", True)
    monkeypatch.setattr(server_mod, "Server", _FakeServer)
    monkeypatch.setattr(server_mod, "TextContent", _FakeTextContent)
    monkeypatch.setattr(server_mod, "discover_read_only_tools", lambda: tools)
    return server_mod.build_server()


def test_call_tool_validates_and_coerces_arguments(monkeypatch):
    seen = {}

    def sample(value: int, output_dir: Path) -> dict:
        seen["value"] = value
        seen["output_dir"] = output_dir
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})
    result = asyncio.run(
        fake_server.call_handler("demo_run", {"value": "7", "output_dir": "out"})
    )

    assert seen["value"] == 7
    assert seen["output_dir"] == Path("out")
    assert json.loads(result[0].text) == {"ok": True}


def test_list_tools_exposes_schema_and_dot_mapping(monkeypatch):
    def sample(value: int, output_dir: Path) -> dict:
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})
    tools = asyncio.run(fake_server.list_handler())

    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "demo_run"
    assert tool.inputSchema["additionalProperties"] is False
    assert tool.inputSchema["properties"]["value"]["type"] == "integer"
    assert tool.inputSchema["properties"]["output_dir"]["type"] == "string"


def test_list_tools_exposes_typed_dict_value_schema(monkeypatch):
    def sample(config: dict[str, int]) -> dict:
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})
    tools = asyncio.run(fake_server.list_handler())

    schema = tools[0].inputSchema["properties"]["config"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"]["type"] == "integer"


def test_list_tools_exposes_any_typed_container_schema(monkeypatch):
    from typing import Any

    def sample(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict:
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})
    tools = asyncio.run(fake_server.list_handler())

    config_schema = tools[0].inputSchema["properties"]["config"]
    rows_schema = tools[0].inputSchema["properties"]["rows"]
    assert config_schema["type"] == "object"
    assert config_schema["additionalProperties"] == {}
    assert rows_schema["type"] == "array"
    assert rows_schema["items"]["type"] == "object"
    assert rows_schema["items"]["additionalProperties"] == {}


def test_fn_to_tool_schema_handles_live_set_and_literal_annotations():
    from mcp_server.server import _fn_to_tool_schema
    from perfxpert.tools import counters, profiling, sol

    profiling_tool = _fn_to_tool_schema("profiling.fill_gap", profiling.fill_gap)
    fingerprint_schema = profiling_tool.inputSchema["properties"]["current_fingerprint"]
    assert fingerprint_schema["type"] == "array"
    assert fingerprint_schema["uniqueItems"] is True
    assert fingerprint_schema["items"]["type"] == "string"

    sol_tool = _fn_to_tool_schema("sol.sanity_check", sol.sanity_check)
    kernel_type_schema = sol_tool.inputSchema["properties"]["kernel_type"]
    assert kernel_type_schema["type"] == "string"
    assert kernel_type_schema["enum"] == ["fp64", "fp32", "bf16"]

    counters_tool = _fn_to_tool_schema("counters.lookup_info", counters.lookup_info)
    gfx_schema = counters_tool.inputSchema["properties"]["gfx_id"]
    assert "anyOf" in gfx_schema
    assert {entry["type"] for entry in gfx_schema["anyOf"]} == {"string", "null"}


def test_call_tool_rejects_invalid_argument_types(monkeypatch):
    called = False

    def sample(value: int) -> dict:
        nonlocal called
        called = True
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})

    with pytest.raises(ValueError, match="invalid value for value"):
        asyncio.run(fake_server.call_handler("demo_run", {"value": "not-an-int"}))
    assert called is False


def test_call_tool_allows_omitted_none_defaults(monkeypatch):
    from perfxpert.tools import compiler, counters

    fake_server = _build_fake_server(
        monkeypatch,
        {
            "compiler.lookup_flags": compiler.lookup_flags,
            "counters.lookup_info": counters.lookup_info,
        },
    )
    flags_result = asyncio.run(fake_server.call_handler("compiler_lookup_flags", {}))
    counter_result = asyncio.run(
        fake_server.call_handler("counters_lookup_info", {"name": "SQ_WAVES"})
    )

    assert isinstance(json.loads(flags_result[0].text), list)
    assert json.loads(counter_result[0].text)["name"] == "SQ_WAVES"


def test_call_tool_rejects_explicit_null_for_required_argument(monkeypatch):
    def sample(value: int) -> dict:
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})

    with pytest.raises(ValueError, match="invalid value for value"):
        asyncio.run(fake_server.call_handler("demo_run", {"value": None}))


def test_call_tool_rejects_invalid_typed_dict_values(monkeypatch):
    def sample(config: dict[str, int]) -> dict:
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})

    with pytest.raises(ValueError, match="invalid value for config"):
        asyncio.run(fake_server.call_handler("demo_run", {"config": {"a": "x"}}))


def test_call_tool_accepts_any_typed_containers(monkeypatch):
    from typing import Any

    def sample(config: dict[str, Any], rows: list[dict[str, Any]]) -> dict:
        return {"config": config, "rows": rows}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})
    result = asyncio.run(
        fake_server.call_handler(
            "demo_run",
            {"config": {"a": 1, "b": {"nested": True}}, "rows": [{"x": 1}, {"y": ["z"]}]},
        )
    )

    payload = json.loads(result[0].text)
    assert payload["config"]["a"] == 1
    assert payload["config"]["b"]["nested"] is True
    assert payload["rows"][1]["y"] == ["z"]


def test_call_tool_rejects_unexpected_arguments(monkeypatch):
    called = False

    def sample(value: int) -> dict:
        nonlocal called
        called = True
        return {"ok": True}

    fake_server = _build_fake_server(monkeypatch, {"demo.run": sample})

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        asyncio.run(fake_server.call_handler("demo_run", {"value": 1, "extra": 2}))
    assert called is False
