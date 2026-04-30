# Architecture: Deterministic Foundation

This layer corresponds to the `Deterministic tools + validated knowledge YAMLs`
node in the main README architecture diagram. It is the non-LLM foundation used
by every entry surface and every agent path.

## Responsibilities

- Own classifier, counter, hardware-fact, trace-diff, roofline, and schema
  logic that must stay deterministic and testable.
- Keep curated YAML knowledge, proven optimization facts, fixtures, and GPU
  architecture data reviewable outside any model prompt.
- Provide stable facts to the agent runtime and correctness middleware so
  agents do not invent counter values, hardware limits, or speedup claims.

## Guides

| Topic | Doc |
|-------|-----|
| Deterministic tool implementation and review rules | [../../contributing/agent-brain/tools.md](../../contributing/agent-brain/tools.md) |
| Validated knowledge YAMLs and update rules | [../../contributing/agent-brain/knowledge.md](../../contributing/agent-brain/knowledge.md) |
| GPU architecture facts | [../../contributing/agent-brain/gpu_arch.md](../../contributing/agent-brain/gpu_arch.md) |
| Proven optimization records | [../../contributing/agent-brain/proven_optimizations.md](../../contributing/agent-brain/proven_optimizations.md) |
| Schemas and fixture coverage | [../../contributing/agent-brain/schemas.md](../../contributing/agent-brain/schemas.md) |

## Related Docs

- [../agent-runtime/](../agent-runtime/) — agents consume these deterministic
  facts when classifying traces or recommending changes.
- [../correctness/](../correctness/) — the gate cascade uses deterministic
  validation before accepting a proposed optimization.
- [../../contributing/agent-brain/](../../contributing/agent-brain/) —
  contributor map for changing agents, tools, knowledge, schemas, and fixtures.
