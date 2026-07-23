# Emulation Agent Skills

This directory uses the open [Agent Skills](https://agentskills.io/)
`SKILL.md` format so compatible Claude, OpenAI, GitHub Copilot, and other agent
clients can share the same workflows.

| Skill | Use it for |
| --- | --- |
| [`emulation-review`](emulation-review/SKILL.md) | Evidence-driven review of an emulation PR, branch, working tree, or file set |
| [`rocjitsu-kernel-parity`](rocjitsu-kernel-parity/SKILL.md) | Focused comparison of rocjitsu KFD/AMDGPU behavior with Linux |
| [`emulation-rebase`](emulation-rebase/SKILL.md) | Rebase or rebuild a branch when some commits or equivalent patches already landed |

The nearest [`AGENTS.md`](../../AGENTS.md) contains shared style, testing, and
confidentiality rules. In particular, `rocjitsu/docs/style.md` is authoritative
for rocjitsu style.

## Optional local references

Reviewers may use a Linux source checkout at `~/linux`. Confidential reference
PDFs may be available at `~/referance` (intentional spelling). Confidential
material must remain local: do not quote, name, link, commit, summarize, or
expose it. Reports must be supportable using repository code, public sources,
or reproducible tests.

The skills do not require a particular agent runner or model. Clients that do
not automatically discover nested `.agents/skills` directories can be pointed
at this directory as a project skill source.
