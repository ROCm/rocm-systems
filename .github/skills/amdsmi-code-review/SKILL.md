---
name: amdsmi-code-review
description: Review pull requests that change files under projects/amdsmi/ (AMD SMI) with the amd-smi review agents. Use for every code review of amd-smi changes.
---

# amd-smi code review

When a pull request changes files under `projects/amdsmi/`, review it with the amd-smi review agents:

1. Read every `amdsmi-review*.agent.md` file in this skill's `agents/` directory.
2. Apply each agent's review checks to the changes under `projects/amdsmi/`. Skip their instructions about tools, subagents, builds and output format; those apply only when they run as VS Code agents.
