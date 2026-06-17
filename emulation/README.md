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
  - [Analysis Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/analysis/) — Register liveness, def-use chains, hazard scheduling, lane permutation. See [liveness.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/analysis/liveness.cpp) and [def_use_chain.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/analysis/def_use_chain.cpp).
  - [ISA Layer](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/) — Per-ISA decoders, operand types, machine instruction structs, encoding formats (CDNA1–4, RDNA1–4, RDNA3.5). Entry points: [decoder.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/decoder.cpp), [rj_decode.cpp](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/rj_decode.cpp); per-arch tables in [arch/](emulation/rocjitsu/lib/rocjitsu/src/rocjitsu/isa/arch/).
  - [simdojo](emulation/rocjitsu/lib/simdojo/) — Decoupled parallel discrete-event simulation engine that orchestrates composable models. See [DESIGN.md](emulation/rocjitsu/lib/simdojo/DESIGN.md) and [src/](emulation/rocjitsu/lib/simdojo/src/) ([simulation.cpp](emulation/rocjitsu/lib/simdojo/src/simulation.cpp), [topology.cpp](emulation/rocjitsu/lib/simdojo/src/topology.cpp), [component.cpp](emulation/rocjitsu/lib/simdojo/src/component.cpp)).
  - [amdisa](emulation/rocjitsu/lib/python/amdisa/) — Python codegen that consumes AMD machine-readable ISA specs to auto-generate decoders, semantics, encoding/legalization tables, and cross-ISA translators. Key modules: [parser.py](emulation/rocjitsu/lib/python/amdisa/parser.py), [semantics.py](emulation/rocjitsu/lib/python/amdisa/semantics.py), [encoding_translator_codegen.py](emulation/rocjitsu/lib/python/amdisa/encoding_translator_codegen.py), [legalization_codegen.py](emulation/rocjitsu/lib/python/amdisa/legalization_codegen.py), [cross_isa.py](emulation/rocjitsu/lib/python/amdisa/cross_isa.py), [codegen/](emulation/rocjitsu/lib/python/amdisa/codegen/).
  - [rocjitsu_vllm](emulation/rocjitsu/lib/python/rocjitsu_vllm/) — Python integration glue for running vLLM on top of rocjitsu.
  - [configs](emulation/rocjitsu/configs/), [schemas](emulation/rocjitsu/schemas/), [scripts](emulation/rocjitsu/scripts/), [tests](emulation/rocjitsu/tests/) — Topology/profile configs, schema definitions, build/dev scripts, and test suites.
- **Mirage** — iOS-simulator-inspired UX and CLI that drives rocjitsu (topology editor, session dashboard, interactive terminal). Code lives in a separate repository; see the [Mirage](#mirage) section below for the architecture.

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
                I1["Per-ISA: decoders, execution, operand types, machine instr structs, encoding formats<br/>9 ISAs: CDNA1–4, RDNA1–4, RDNA3.5"]
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

##

## Mirage
An IOS simulator inspired UX and CLI to make it easy to use rocjitsu.

```mermaid
flowchart TB
    subgraph MIRAGE["Mirage"]
        subgraph UI["UI"]
            TOPOE["Topology Editor"]
            TERM["Interactive Terminal"]
            DASHBOARD["Session Dashboard"]
        end

        TOPO["Topology"]
        
        SIM["Emulator"]
        PLUGINS["Plugins"]
        TOPO --> PROFILE
        SIM --> PROFILE
        PLUGINS --> PROFILE
        PROFILE["Profile"]
        SESSION["Session"]
        METRICS["Metrics"]
        SESSION --> METRICS
        METRICS --> DASHBOARD 
        EXECUTION["Execution "]
        PROFILE --> SESSION --> EXECUTION
        SESSION --> DASHBOARD
        CONTAINERS["Containers"]
        EXECUTION --> CONTAINERS
        EXECUTION --> SIM
        EXECUTION --> TERM
        TERM --> EXECUTION
        SIM --> METRICS
        TOPOE --> TOPO
    end
```

### Server

Hosts the UI server
Written in Rust
Designed to be idomorphics so can be started and stopped whenever.
All state is stored to disc.

| term | definition |
| --- | --- |
| profile | a config |
| session | a booted running profile |
| execution | running a program on a session |
| workload | a profile + execution |

### Dashboard

A react dashboard.

#### Topology Editor

Allows creating custom gpus

### Session

view session state + metrics.
start executions

#### Executions

Interactive terminals for sessions