# Governance

## Core reviewers

People who can approve standard PRs (tools, knowledge, fixtures, agents, docs):

- `@aelwazir` (AMD, project lead)
- `@rocm-maintainer-1` (AMD, performance domain)
- `@rocm-maintainer-2` (AMD, agents/LLM domain)

## Security-focused reviewers

People who audit provider and MCP-tool changes:

- `@security-lead` (AMD, security)

## Architectural reviewers (3-person gate)

Required for RFCs that touch correctness gates, shared fence, or agent tree:

- `@aelwazir`
- `@rocm-maintainer-1`
- `@rocm-maintainer-2`

## Review SLA

- Standard PR: 2 business days
- Architectural (RFC): 5 business days (includes 1-week discussion + voting)
- Security audit (provider/MCP): 3 business days

## Escalation

If a PR stalls:
1. Comment with `@aelwazir` + context
2. Project lead will either review or delegate within 24h
3. If lead unavailable, escalate to AMD ROCm team leads

## Governance change policy

Changes to this file (adding/removing reviewers, changing SLAs) require:
- **3 core maintainer approvals** (architectural change to the project)
- **1-week discussion period**
- **RFC in `docs/rfcs/`** with rationale

This ensures continuity and prevents reviewer churn.

## Decision precedent

In case of reviewer disagreement:
1. Try to reach consensus in the PR thread
2. If no consensus after 3 business days, project lead (`@aelwazir`) makes final call
3. Rationale recorded in commit message or PR comment

---

## Related docs

- `../CONTRIBUTING.md` (reviewer requirements table)
- `../rfcs/` (RFC process for architectural changes)
