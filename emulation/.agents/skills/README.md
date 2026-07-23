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

Reviewers may use a Linux source checkout at `~/linux`. Users should put public
shader programming guides, ISA manuals, and architecture documents in
`~/reference/public/shader-programming-guides`. Public copies can be obtained
from
[AMD GPU architecture programming documentation](https://gpuopen.com/amd-gpu-architecture-programming-documentation/).
Keep confidential PDFs separately in `~/reference/confidential`.

Confidential material and its contents must remain local. Skills and reviewers
must not quote, name, link, copy, commit, summarize, upload, or expose it through
paths, metadata, screenshots, logs, prompts, generated artifacts, internal
terminology, issues, pull requests, reviews, tests, or chat. Reports must be
supportable using repository code, public sources, or reproducible tests. Treat
material of uncertain publication status as confidential.

The skills do not require a particular agent runner or model. Clients that do
not automatically discover nested `.agents/skills` directories can be pointed
at this directory as a project skill source.

## GitHub write approval

Every skill must obtain explicit user approval immediately before each GitHub
write, including pushes and creating or changing pull requests, issues,
comments, reviews, labels, tags, releases, or other remote state. Earlier task
approval and approval for a different write do not count. Read-only GitHub
inspection is allowed; publishing is not implicit in review, rebase, commit, or
pull-request preparation requests.
