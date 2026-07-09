# AMD-SMI Agent Flow

Top-down view of how a user request travels through the agent system: the
**Planning agent** is the lead orchestrator and entry point. It triages intent,
then either answers inline, dispatches **Explore** for read-only investigation,
routes straight to **Development** or **Review**, or owns the goal end-to-end
through the plan → dev↔review loop. Solid arrows are sequential skill flow within
an agent; dashed arrows are agent-to-agent dispatch (Planning calls down into the
shared Development and Review agents).

```mermaid
flowchart TD
    User([User request])
    User --> Plan{{Planning Agent · lead orchestrator}}

    Plan --> Dev[Development Agent]
    Plan --> Rev[Review Agent]
    Plan --> Explore[Explore · read-only]
    Plan --> Inline[Inline answer<br/>trivial only]

    %% ---- Planning column ----
    Plan --> P1[interrogate<br/>→ approved spec]
    P1 --> P2[writing-plans<br/>→ bite-sized plan]
    P2 --> P3[iterate per task]
    P3 --> P4{verification<br/>1 · plan complete<br/>2 · spec satisfied<br/>3 · review clean}
    P4 -->|any NO → next iteration| P3
    P4 -->|all YES| P5[restructure-commits<br/>→ finish]

    %% ---- Development column ----
    Dev --> D1[interrogate + plan<br/>direct mode only]
    D1 --> D2[test-driven-development<br/>RED → GREEN → REFACTOR]
    D2 --> D3[systematic-debugging<br/>on failures]
    D3 --> D4[verification-before-completion]

    %% ---- Review column ----
    Rev --> V1[build + style<br/>always-on]
    V1 --> V2[tests · docs · arch<br/>security · perf · spec]
    V2 --> V3[skeptic + rebuttal]
    V3 --> V4[findings + status]

    %% ---- real dispatch (Planning calls down into the shared agents) ----
    P3 -.dispatch dev task.-> Dev
    P3 -.dispatch review.-> Rev

    classDef apex fill:#1f6feb,color:#fff,stroke:#0d47a1,stroke-width:2px
    classDef tier fill:#2d333b,color:#fff,stroke:#444c56
    classDef gate fill:#b3541e,color:#fff,stroke:#7a3812
    class Plan apex
    class Dev,Rev tier
    class P4 gate
```

**Verification gate (Planning):** all three must be YES to finish —
1. **Plan complete** — every plan task has a committed change.
2. **Spec satisfied** — every spec requirement is observable (cascade grep + smoke commands).
3. **Review clean** — no ❌ BLOCKING findings in the most recent comprehensive review.
