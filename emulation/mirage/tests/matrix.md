# mirage end-to-end testing matrix

This document defines the end-to-end (E2E) test matrix for `mirage`. It
describes the lifecycle each combination is driven through, the
dimensions that are combined, and the policy for skipping combinations
that the host cannot run.

The matrix is implemented by
[`tests/matrix_e2e.rs`](./matrix_e2e.rs), which enumerates every
combination below and exercises it against the real `mirage` binary
under an isolated XDG root. Combinations that the current host cannot
support are **skipped with a recorded reason** rather than failed, so
the same suite is meaningful on a developer laptop, in CI, and on an
emulation host.

## Lifecycle

Every runnable combination is driven through the canonical session
lifecycle and asserted at each step:

1. **Create** — create a profile pinned to the combination's emulator,
   agent, node count, and (optional) container settings.
2. **Run** — execute the payload via `mirage run`, which creates a
   transient session, runs the workload on every node, and asserts the
   expected exit status.
3. **Delete** — remove the profile.
4. **Ensure deleted** — confirm the transient session cleaned itself up
   (no session directories remain) and that the profile is gone.

## Dimensions

The matrix is the full cross product of the following dimensions.

### Emulator

| Value          | Description                                                                |
| -------------- | -------------------------------------------------------------------------- |
| `rocjitsu`     | Software GPU emulator. Runs on any host once its KMD library is installed. |
| `rocjitsu-dbt` | Dynamic binary translation. Runs translated code on a **physical** GPU.    |

### Containerization

| Value    | Description                                              |
| -------- | -------------------------------------------------------- |
| `node`   | Run directly on the node; no container runtime involved. |
| `podman` | Run each node inside a Podman container.                 |
| `docker` | Run each node inside a Docker container.                 |

### Hardware (emulated GPU)

| Value    | Agent    |
| -------- | -------- |
| `mi350x` | `MI350X` |
| `mi450x` | `MI450X` |

### Payload

| Value        | Nodes | Description                                                     |
| ------------ | ----- | -------------------------------------------------------------- |
| `tiny_torch` | 1     | A minimal Torch-style workload (single node).                  |
| `rccl`       | 2     | A multi-node collective workload (two nodes).                  |
| `crash`      | 1     | A workload that exits abnormally; verifies cleanup still runs. |

### Plugins

| Value              | Description                     |
| ------------------ | ------------------------------- |
| `none`             | No emulator plugins.            |
| `hazard-detection` | Memory-hazard detection plugin. |

## Skip policy

A combination is **skipped** (not failed) when the host cannot run it.
The reason is recorded in the test output. The following rules apply:

* **`rocjitsu-dbt` without a translation-target GPU** — DBT executes
  translated code on real hardware, so it is skipped on any host without
  a supported physical GPU. This is the primary hardware-gated skip.
* **`rocjitsu-dbt` with an `mi450x` guest** — `gfx1250` is not a
  DBT-translatable source ISA, so this guest is skipped even when GPU
  hardware is present.
* **`rocjitsu` without its KMD library** — the software emulator is
  skipped when `mirage` reports it as not installed (its KMD library
  could not be located by mirage's own discovery).
* **`hazard-detection` when unavailable** — skipped when the active
  backend does not advertise the plugin.

The suite asserts that at least one combination runs whenever the
software emulator is installed, so an environment misconfiguration cannot
silently turn the matrix into a no-op.

## Running

Run the matrix as part of the normal test suite:

```sh
cargo test --test matrix_e2e -- --nocapture
```

`--nocapture` prints the per-combination `RAN` / `SKIP` table and a
summary line (`N ran, M skipped, T total`).

The `rocjitsu` rows run whenever `mirage` discovers its KMD library; see
[`building.md`](../docs/building.md) for the discovery order (a sibling
monorepo build, `$ROCM_HOME/lib`, or `$(rocm-sdk path --root)/lib`).

The containerized dimensions (`podman`, `docker`) are driven through a
hermetic mock provider — a small shell script standing in for the
container CLI — so the provider bring-up and teardown contract is
exercised without requiring a real image or daemon. This mirrors
[`tests/container_e2e.rs`](./container_e2e.rs).

## Related

* [`tests/run_tiny_torch_mi350.sh`](./run_tiny_torch_mi350.sh) — runs the
  real Torch fixture on an emulated MI350X using a ROCm nightly venv.
* [`tests/container_e2e.rs`](./container_e2e.rs) — containerized
  lifecycle against the mock provider.
* [`tests/lifecycle_e2e.rs`](./lifecycle_e2e.rs) — file-driven signal and
  cleanup pathways.
