# Contributing: new knowledge entry

## What you're adding

A structured YAML file containing GPU specs, performance heuristics, counter
catalogs, or other reference data. Knowledge is the immutable reference layer
that tools and agents consult without executing LLM calls.

## File locations

- Data: `perfxpert/knowledge/<name>.yaml`
- JSON schema: `perfxpert/knowledge/_schemas/<name>.schema.json`
- Tests: `tests/test_knowledge/test_<name>.py`

## Template

### YAML skeleton

```yaml
# perfxpert/knowledge/<name>.yaml
version: "1.0"
metadata:
  description: "One-sentence description"
  maintained_by: "your-handle or team"
  last_updated: "2026-04-18"

entries:
  - id: entry_1
    field: "value"
  - id: entry_2
    field: "value"
```

### JSON schema skeleton

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "version": {"type": "string"},
    "metadata": {
      "type": "object",
      "properties": {
        "description": {"type": "string"},
        "maintained_by": {"type": "string"},
        "last_updated": {"type": "string"}
      },
      "required": ["description"]
    },
    "entries": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {"type": "string"},
          "field": {"type": "string"}
        },
        "required": ["id"]
      }
    }
  },
  "required": ["entries"]
}
```

## Schema constraints (CI-enforced)

- Every YAML has a corresponding `.schema.json` in `_schemas/`
- `test_yaml_schemas.py` validates all YAML files against their schemas on every PR
- Schema follows JSON Schema draft-07
- YAML load must pass pyyaml (no circular refs, no code injection)

## Tests you must add

- `test_<name>_loads_without_error()` — happy path
- `test_<name>_validates_against_schema()` — schema conformance
- `test_<name>_has_required_fields()` — spot-check key entries
- `test_<name>_ids_are_unique()` — no duplicate IDs if applicable

## Review requirements

- 1 core reviewer
- CI green (schema + load validation)

## Common pitfalls

- Schema must be a valid JSON Schema draft-07 (not hand-wavy)
- YAML IDs used by tools/agents for lookup — keep them stable
- If you edit an existing YAML, schema might also need to change (version bump + backcompat note)
- Comments in YAML are allowed; comments in JSON schemas are not

## Related docs

- Existing knowledge under `perfxpert/knowledge/` as references
- JSON Schema handbook: https://json-schema.org/
- test_yaml_schemas.py for validation patterns
