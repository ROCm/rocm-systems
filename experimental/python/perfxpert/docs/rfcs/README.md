# perfxpert RFC Process

Architectural-scope changes go through an RFC in this directory. Smaller
changes (new tool, new knowledge entry) do NOT need an RFC — see
`../CONTRIBUTING.md` governance table.

## When an RFC is required

- Changing the 5-gate correctness cascade logic (§5)
- Editing the shared fence layer (`agents/fence/always.md`)
- Adding / removing an agent from the top-level tree
- Changing the tool-class split rules (READ_ONLY vs EXECUTION)
- Changing the provider `Model` protocol
- Changing the MCP exposure policy
- Changing the public Python API in a breaking way

## Process

1. **Draft:** copy `TEMPLATE.md` to `NNNN-short-title.md`, fill in sections
2. **Open PR:** mark as draft, add `rfc` label
3. **Discussion period:** minimum **1 week** (architectural RFCs: 2 weeks)
4. **Reviewer threshold:**
   - Correctness-gate / `always.md` / public-API RFC: **3 core maintainers** approve
   - Fence / schema / single-agent RFC: **2 core maintainers** approve
   - Provider / MCP / security RFC: **1 security-focused maintainer** + **1 core maintainer** approve
5. **Merge:** RFC merged becomes design of record; supersedes any prior
   design-doc content on the same topic
6. **Implementation:** separate PRs reference the RFC number in commits
   (`Implements: RFC-0042`)

## Withdrawal

An RFC can be withdrawn at any time before merge — edit status in
front-matter to `withdrawn`, leave the file in place as a record.

## Numbering

- IDs are monotonically increasing (next = highest + 1)
- `0000-example-rfc.md` is a canonical demo and does not count

## Index

_Populated as RFCs are merged._
