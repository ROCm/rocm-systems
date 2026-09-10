# ROCM Emulation

[work board](https://github.com/orgs/ROCm/projects/164)

## Components

- [Rocjitsu](emulation/rocjitsu/) — JIT-based AMD GPU emulator. The core engine that interprets/translates AMDGPU ISA and models GPU blocks so unmodified ROCm apps can run without a physical GPU.
  - [KMD Interposer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/kmd/) — Intercepts kernel-mode driver calls and routes them to the simulated KMD. Per-OS backends in [linux/](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/kmd/linux/) and [windows/](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/kmd/windows/).
  - [Config Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/config/) — Loads and composes topology/profile configs (e.g. MI350X) on top of the modular VM. See [config_loader.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/config/config_loader.cpp) and [checkpoint.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/config/checkpoint.cpp).
  - [VM Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/) — Virtual machine modeling GPU blocks, SoC, and thread context; drives instruction execution. Plugin harness via [execution_plugin.h](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/execution_plugin.h). Per-arch backends in [amdgpu/](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/amdgpu/) and [risc_v/](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/risc_v/). Design notes: [DESIGN.md](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/vm/DESIGN.md).
  - [Code Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/) — Code object APIs: ELF parse/patch, basic blocks, kernel descriptors, executables. See [amdgpu_code_object.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/amdgpu_code_object.cpp), [basic_block.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/basic_block.cpp), [executable.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/executable.cpp).
    - [DBT](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/dbt/) — Dynamic binary translation: encoding and semantic translation.
    - [Patch / DBI](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/patch/) — Patch points and (planned) instrumentation, spill/restore.
  - [Analysis Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/analysis/) — Register liveness, def-use chains, hazard scheduling, lane permutation. See [liveness.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/analysis/liveness.cpp) and [def_use_chain.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/code/analysis/def_use_chain.cpp).
  - [ISA Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/) — Per-ISA decoders, operand types, machine instruction structs, encoding formats (CDNA1–5, RDNA1–4). Entry points: [decoder.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/decoder.cpp), [rj_decode.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/rj_decode.cpp); per-arch tables in [arch/](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/).
  - [simdojo](emulation/rocjitsu/lib/simdojo/) — Decoupled parallel discrete-event simulation engine that orchestrates composable models. See [DESIGN.md](emulation/rocjitsu/lib/simdojo/DESIGN.md) and [src/](emulation/rocjitsu/lib/simdojo/src/) ([simulation.cpp](emulation/rocjitsu/lib/simdojo/src/simulation.cpp), [topology.cpp](emulation/rocjitsu/lib/simdojo/src/topology.cpp), [component.cpp](emulation/rocjitsu/lib/simdojo/src/component.cpp)).
  - [amdisa](emulation/rocjitsu/lib/python/amdisa/) — Python codegen that consumes AMD machine-readable ISA specs to auto-generate decoders, semantics, encoding/legalization tables, and cross-ISA translators. Key modules: [parser.py](emulation/rocjitsu/lib/python/amdisa/parser.py), [semantics.py](emulation/rocjitsu/lib/python/amdisa/semantics.py), [encoding_translator_codegen.py](emulation/rocjitsu/lib/python/amdisa/encoding_translator_codegen.py), [legalization_codegen.py](emulation/rocjitsu/lib/python/amdisa/legalization_codegen.py), [cross_isa.py](emulation/rocjitsu/lib/python/amdisa/cross_isa.py), [codegen/](emulation/rocjitsu/lib/python/amdisa/codegen/).
  - [rocjitsu_vllm](emulation/rocjitsu/lib/python/rocjitsu_vllm/) — Python integration glue for running vLLM on top of rocjitsu.
  - [configs](emulation/rocjitsu/configs/), [schemas](emulation/rocjitsu/schemas/), [scripts](emulation/rocjitsu/scripts/), [tests](emulation/rocjitsu/tests/) — Topology/profile configs, schema definitions, build/dev scripts, and test suites.
- [CLI](emulation/rocjitsu/cli/) — iOS-simulator-inspired UX that drives the emulator (named profiles, persistent sessions, containers). A Rust workspace whose top-level [`rocjitsu`](emulation/rocjitsu/cli/src/) binary composes the crates below. See the [CLI](#cli) section for the architecture.
  - [core](emulation/rocjitsu/cli/core/) (`rj_core`) — Shared library: profile/session/topology/agent stores, the control-plane trait, KMD/library discovery, and container env wiring. See [discovery.rs](emulation/rocjitsu/cli/core/src/discovery.rs), [profile.rs](emulation/rocjitsu/cli/core/src/profile.rs), [ctl.rs](emulation/rocjitsu/cli/core/src/ctl.rs), [container.rs](emulation/rocjitsu/cli/core/src/container.rs).
  - [ctl](emulation/rocjitsu/cli/ctl/) (`rj_ctl`) — User-facing control plane: parses and dispatches every `rocjitsu` subcommand (profile/topology/agent/session/exec/run/…). Entry point [lib.rs](emulation/rocjitsu/cli/ctl/src/lib.rs).
  - [supervisor](emulation/rocjitsu/cli/supervisor/) (`rj_supervisor`) — Per-session supervisor that boots a session, brings up containers, hosts the emulator daemon, and runs/streams executions across nodes and ranks. See [session.rs](emulation/rocjitsu/cli/supervisor/src/session.rs) and [run.rs](emulation/rocjitsu/cli/supervisor/src/run.rs).
  - [container](emulation/rocjitsu/cli/container/) (`rj_container`) — Drives the `docker`/`podman` CLI to realise the containerised parts of a session: image build/pull, networks, node launch. See [lib.rs](emulation/rocjitsu/cli/container/src/lib.rs).
  - [builtin](emulation/rocjitsu/cli/builtin/) (`rj_builtin`) — Built-in agents and topologies (e.g. MI300X/MI350X) preloaded into the user config. See [agents.rs](emulation/rocjitsu/cli/builtin/src/agents.rs) and [profiles.rs](emulation/rocjitsu/cli/builtin/src/profiles.rs).
  - Emulator backends — pluggable engines registered via `inventory`, selected per profile by `kind`:
    - [rocjitsu](emulation/rocjitsu/cli/rocjitsu/) (`rj_backend_rocjitsu`) — Emulator integration: synthesizes the sim config, hosts the emulator daemon, and the DBT translation and guest backends. See [lib.rs](emulation/rocjitsu/cli/rocjitsu/src/lib.rs), [dbt.rs](emulation/rocjitsu/cli/rocjitsu/src/dbt.rs), [guest.rs](emulation/rocjitsu/cli/rocjitsu/src/guest.rs).
    - [hotswap](emulation/rocjitsu/cli/hotswap/) (`rj_backend_hotswap`) — HotSwap backend: load-time ISA rewriter that runs a guest-built workload on the host GPU. See [lib.rs](emulation/rocjitsu/cli/hotswap/src/lib.rs).
  - [rocjitsu_sys](emulation/rocjitsu/cli/rocjitsu_sys/) — Runtime (dlopen) FFI bindings to the emulator's C API (`rj_vm_*`), and the one crate in the workspace permitted to write `unsafe`. See [lib.rs](emulation/rocjitsu/cli/rocjitsu_sys/src/lib.rs).
  - [scripts](emulation/rocjitsu/cli/scripts/), [tests](emulation/rocjitsu/cli/tests/), [docs](emulation/rocjitsu/cli/docs/), [cmake](emulation/rocjitsu/cli/cmake/) — Build/dev and Dockerized-build scripts, end-to-end and ML test fixtures, design docs, and the CMake wrapper over cargo.

## Rocjitsu

The heart and core of the emulation effort.

```mermaid
flowchart TB
    subgraph APPS["Applications & Frameworks (unmodified)"]
        A1["PyTorch"]
        A2["vLLM"]
        A3["rocBLAS"]
        A4["HIP kernels"]
    end

    subgraph ROCM["ROCm Stack (unmodified)"]
        R1["HIP<br/>(High-level GPU API)"]
        R2["HSA<br/>(Heterogeneous System Architecture)"]
        R3["ROCR<br/>(ROCm Runtime)"]
        R1 --> R2 --> R3
    end
    A1 --> R1
    A2 --> R1
    A3 --> R1
    A4 --> R1

    subgraph CORE["Rocjitsu"]

        subgraph KMD["KMD Interposer"]
            K1["Intercepts driver calls, routes to simulated kernel-mode driver.<br/>Runs unmodified apps."]
        end

        subgraph CFG["Config Layer"]
            C1["Compose specific topologies<br/>(e.g., MI350X) on top of<br/>the modular VM."]
        end
        subgraph SIM["simdojo (PDES engine)"]
            direction LR
            S1["Decoupled parallel discrete event simulation engine<br/>orchestrates scalable simulation and composable models"]
            subgraph VM["VM Layer"]
                direction TB
                V1["Virtual machine models GPU blocks<br/>and context, drives instruction execution."]
                subgraph PLUGINS["Plugins"]
                    direction TB
                    P1["loading of tools"]
                    P2["Hazard detection"]
                    PM["Metrics"]
                    P3["Timing Model"]
                    P1 --> P2 --> PM
                    P1 --> P3 --> PM
                end
            end
            subgraph CODE["Code Layer"]
                direction TB
                subgraph CODEAPI["Code Object APIs"]
                    CO1["ELF parse/patch"]
                    CO2["Basic blocks"]
                    CO3["Kernel descriptors"]
                end
                subgraph DBT["DBT"]
                    D1["Encoding translation"]
                    D1b["Semantic translation"]
                end
                subgraph DBI["DBI (planned)"]
                    D2["Instrumentation"]
                    D3["Spill/restore"]
                    D4["Patch points"]
                end
                subgraph ANA["Analysis Layer"]
                    AN1["Register liveness"]
                    AN2["Hazard scheduling"]
                    AN3["Lane permutation"]
                end
                CODEAPI --> DBT --> ANA
                CODEAPI --> DBI --> ANA
            end
            subgraph ISA["ISA Layer"]
                I1["Per-ISA: decoders, execution, operand types, machine instr structs, encoding formats<br/>CDNA1–5, RDNA1–4"]
                subgraph AMDISA["amdisa (Python codegen)"]
                    AM1["Leverages the AMD machine-readable ISA specs to auto-generate<br/>ISA description and semantics for simulation, instrumentation, and translation"]
                end
            end
            CFG --> VM --> CODE --> ISA
        end
    end


    R3 --> KMD --> SIM

    classDef apps fill:#e8eef7,stroke:#5b7aa8
    classDef kmd fill:#f3e2ec,stroke:#a8567a
    classDef cfg fill:#eeeae0,stroke:#8a7a55
    classDef vm fill:#e6e0f0,stroke:#6b5ba8
    classDef code fill:#f7dede,stroke:#a85b5b
    classDef ana fill:#dceee6,stroke:#4f8a72
    classDef isa fill:#e3efd9,stroke:#6f9a52
    classDef sim fill:#f7e6cf,stroke:#b58440
    classDef amdisa fill:#d9ebe5,stroke:#3f8a78

    class APPS,ROCM apps
    class KMD kmd
    class CFG cfg
    class VM vm
    class CODE,DBT,DBI code
    class ANA ana
    class ISA isa
    class SIM sim
    class AMDISA amdisa
```

## CLI

An iOS-simulator-inspired UX that makes the emulator easy to drive. Where the engine above is the raw emulation core, the CLI is the control plane around it: it turns a one-off `LD_PRELOAD` invocation into a managed, reproducible workflow with named configs, persistent sessions, and containers.

The CLI exists to:

- **Drive the emulator.** A single `rocjitsu run` boots an emulated GPU and launches your unmodified ROCm workload on it — it discovers the KMD interposer, synthesizes the simulation config from your profile, hosts the emulator daemon, and injects the right environment so the app "sees" the virtual hardware.
- **Support complex topologies.** Describe multi-GPU, multi-node systems (e.g. MI300X/MI350X/MI450X) declaratively. The `--num-nodes` / `--gpus-per-node` / `--nproc-per-node` flags let you stand up large virtual clusters, and the CLI wires up rendezvous (`MASTER_ADDR`, `WORLD_SIZE`, `RANK`, per-node `NCCL_HOSTID`, …) so distributed frameworks like torchrun and RCCL run unchanged.
- **Support containers and reproducible runs.** A profile is a self-contained, on-disk description of a workload. Executions run inside containers with pinned images so a run can be replayed identically on another machine, and the CLI handles image bring-up, mounts, and host/glibc compatibility automatically.
- **Make it easier to debug and run.** Persistent sessions, `rocjitsu exec` into a live session from another terminal, and streamed build/boot logs replace the manual dance of preloading libraries and hand-editing JSON configs.

```mermaid
flowchart TB
    subgraph CLI["rocjitsu CLI"]
        TOPO["Topology"]
        SIM["Emulator"]
        PLUGINS["Plugins"]
        PROFILE["Profile"]
        SESSION["Session"]
        EXECUTION["Execution"]
        CONTAINERS["Containers"]
        TOPO --> PROFILE
        SIM --> PROFILE
        PLUGINS --> PROFILE
        PROFILE --> SESSION --> EXECUTION
        EXECUTION --> CONTAINERS
        EXECUTION --> SIM
    end
```

### Concepts

The CLI models a workload as a small set of layered, persistent objects:

| term | definition |
| --- | --- |
| profile | a config: emulator + topology + container/image + plugins |
| session | a booted, running profile (an emulated machine) |
| execution | running a program on a session |
| workload | a profile + execution |
