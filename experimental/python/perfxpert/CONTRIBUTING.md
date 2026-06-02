# Contributing to perfxpert

Thanks for contributing to perfxpert! This doc is the master index. For each
extension surface there's a per-surface guide in `docs/contributing/`.

## TL;DR

```bash
# One-time setup
git clone <repo> && cd rocm-systems/experimental/python/perfxpert
pip install -e '.[dev]'

# Every PR
pytest                      # full test suite — must be green
perfxpert doctor            # health check — must be clean
ruff check perfxpert/       # lint (optional — CI runs it anyway)
```

## Extension surfaces (quick reference)

| Contribution | Where | Guide |
|--------------|-------|-------|
| New tool | `perfxpert/tools/<module>.py` | [docs/contributing/tools.md](docs/contributing/tools.md) |
| New knowledge entry | `perfxpert/knowledge/<name>.yaml` | [docs/contributing/knowledge.md](docs/contributing/knowledge.md) |
| New proven optimization | `perfxpert/knowledge/proven_optimizations.yaml` + fixture pair | [docs/contributing/proven_optimizations.md](docs/contributing/proven_optimizations.md) |
| New agent | `perfxpert/agents/<name>.py` + `agents/fence/<name>.md` | [docs/contributing/agents.md](docs/contributing/agents.md) |
| New LLM provider | `perfxpert/providers/<name>_model.py` | [docs/contributing/providers.md](docs/contributing/providers.md) |
| New MCP tool | `mcp_server/tools/<name>.py` | [docs/contributing/mcp_tools.md](docs/contributing/mcp_tools.md) |
| New test fixture | `tests/fixtures/<name>.db` + `.md` | [docs/contributing/fixtures.md](docs/contributing/fixtures.md) |
| New GPU arch | `knowledge/gpu_specs.yaml` + `vgpr_occupancy_tables.yaml` | [docs/contributing/gpu_arch.md](docs/contributing/gpu_arch.md) |
| External-tool dependency (`require_tool` registration) | `perfxpert/tools/_tooldep.py` | [docs/contributing/external-tools.md](docs/contributing/external-tools.md) |

The per-surface guides are indexed under
[docs/contributing/README.md](docs/contributing/README.md).

## Governance

| Change | Reviewers | RFC required? |
|--------|-----------|---------------|
| Tool / knowledge / fixture / doc | 1 core reviewer | no |
| Proven-optimization entry | 1 core reviewer + CI green | no |
| Agent | 2 core reviewers (architectural) | no |
| Provider / MCP tool | 1 security-focused reviewer | no |
| Correctness-gate logic change | 3 core maintainers | **yes** — `docs/rfcs/` |
| `always.md` shared fence change | 2 core maintainers | **yes** — `docs/rfcs/` |

See [docs/contributing/governance.md](docs/contributing/governance.md) for the full reviewer list + escalation path.

## Narrow-scope discipline

perfxpert enforces narrow scope per agent and per tool (spec §2):
- Fences ≤ 400 lines
- ≤ 5 tools per agent allowlist
- ≤ 10 input fields / ≤ 5 output fields per handoff schema
- Tools are either `READ_ONLY` (MCP-safe) or `EXECUTION` (in-process only)

CI enforces these constraints mechanically (`tests/test_integration/test_narrow_scope.py`).

## Worked example: adding a new bottleneck class

End-to-end walkthrough of adding `io_bound` as a bottleneck class —
covers knowledge, tool, agent, fence, test, fixture in ≤ 7 files without
needing an RFC: [docs/contributing/walkthrough_new_bottleneck_class.md](docs/contributing/walkthrough_new_bottleneck_class.md).

## RFC process

Architectural changes (new gate logic, fence-shared-layer edits, agent
tree restructures) go through `docs/rfcs/`. Process: 1-week discussion
minimum, template in `docs/rfcs/TEMPLATE.md`, merged RFC becomes design of record.

See [docs/rfcs/README.md](docs/rfcs/README.md).

## Test pyramid (spec §6)

```
Level 5 — Benchmarks (nightly)        TritonBench + KernelBench
Level 4 — End-to-end CLI + SDK (PR)   pytest tests/e2e
Level 3 — Agent integration (PR)      pytest tests/test_integration
Level 2 — Per-agent isolation (PR)    pytest tests/test_agents
Level 1 — Per-tool unit (PR)          pytest tests/test_tools
Level 0 — Knowledge YAML (PR)         pytest tests/test_knowledge
```

Every PR must stay green on Levels 0-4. Nightly runs Level 5.

## Code of conduct

perfxpert inherits ROCm Systems' Code of Conduct — see the top-level
`rocm-systems` repository for the canonical text.

## Licensing

perfxpert is MIT-licensed (see LICENSE). Contributions are accepted under
the same terms via the Developer Certificate of Origin — sign off with
`git commit -s` to acknowledge.
