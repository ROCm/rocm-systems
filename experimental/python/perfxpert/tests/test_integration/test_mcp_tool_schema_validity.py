"""Validate every MCP tool's JSON schema for OpenAI function-calling compatibility.

Every tool our MCP server registers must have a JSON schema that:
1. Is a valid JSON Schema Draft 7 document (per MCP spec)
2. Is acceptable as an OpenAI function-calling schema (requires:
   - type: object for inputSchema
   - every array has "items"
   - every object-type property has "additionalProperties" or "properties")

This test catches bugs like "array without items" that opencode rejects.
"""

import jsonschema

from mcp_server._registry import discover_read_only_tools
from mcp_server.server import _fn_to_tool_schema


def _walk_props(properties):
    """Recursively yield (property_name, schema_fragment) tuples."""
    for pname, pschema in properties.items():
        yield pname, pschema
        if pschema.get("type") == "object" and "properties" in pschema:
            yield from _walk_props(pschema["properties"])


def test_every_tool_schema_is_valid_draft7():
    """Validate every tool's inputSchema is valid JSON Schema Draft 7."""
    for name, fn in discover_read_only_tools().items():
        tool = _fn_to_tool_schema(name, fn)
        try:
            jsonschema.Draft7Validator.check_schema(tool.inputSchema)
        except jsonschema.SchemaError as e:
            raise AssertionError(f"tool {name} has invalid schema: {e}")


def test_every_array_property_has_items():
    """Every array property must have an 'items' key (OpenAI requirement).

    This check would have caught the bug where we emitted:
        {"type": "array"}  # INVALID — missing items
    instead of:
        {"type": "array", "items": {...}}  # VALID
    """
    offenders = []
    for name, fn in discover_read_only_tools().items():
        tool = _fn_to_tool_schema(name, fn)
        for pname, pschema in _walk_props(tool.inputSchema.get("properties", {})):
            if pschema.get("type") == "array" and "items" not in pschema:
                offenders.append(f"{name}::{pname}")
    assert not offenders, (
        f"arrays without items (OpenAI rejects): {offenders}"
    )


def test_every_input_schema_has_type_object():
    """Every tool's inputSchema must be type=object (MCP + OpenAI requirement)."""
    for name, fn in discover_read_only_tools().items():
        tool = _fn_to_tool_schema(name, fn)
        assert tool.inputSchema.get("type") == "object", (
            f"{name} inputSchema must be type=object, got {tool.inputSchema.get('type')}"
        )


def test_object_properties_have_additionalproperties():
    """Every object-type property must have additionalProperties set."""
    for name, fn in discover_read_only_tools().items():
        tool = _fn_to_tool_schema(name, fn)
        for pname, pschema in _walk_props(tool.inputSchema.get("properties", {})):
            if pschema.get("type") == "object":
                assert "additionalProperties" in pschema or "properties" in pschema, (
                    f"{name}::{pname} is type=object but missing additionalProperties"
                )
