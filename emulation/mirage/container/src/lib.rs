//! Container engine: drives the `docker`/`podman` CLI to realise the
//! containerised parts of a mirage session.
//!
//! The *static* configuration ([`mirage_core::profile::ContainerizedDef`],
//! [`mirage_core::profile::FileMount`]) and the *runtime* record
//! ([`mirage_core::container::ContainerState`], naming + provider
//! resolution, and dependency-free [`teardown`](mirage_core::container::teardown))
//! live in `mirage_core`. This crate adds the imperative orchestration:
//! pulling images, creating the per-session network, launching one
//! container per node, and building the `exec` argv used to run a
//! command inside a node's container.
//!
//! The design keeps a clean split:
//!
//! * **argv builders** ([`Engine::run_argv`], [`Engine::exec_argv`]) are
//!   pure functions of their inputs and fully unit-tested without a real
//!   runtime.
//! * **side-effecting methods** ([`Engine::pull`], [`Engine::ensure_network`],
//!   [`Engine::launch_node`], …) invoke the provider and are exercised in
//!   tests with a mock provider shell script.

use std::process::{Command, Stdio};

use mirage_core::container::{ContainerState, NodeContainer};
use mirage_core::profile::{ContainerizedDef, FileMount};

/// Errors raised while driving a container provider.
#[derive(Debug, thiserror::Error)]
pub enum ContainerError {
    /// No provider was configured and none could be auto-detected.
    #[error(
        "no container provider found; install podman or docker, or set MIRAGE_CONTAINER_PROVIDER"
    )]
    NoProvider,

    /// The provider binary could not be spawned.
    #[error("failed to spawn `{provider} {}`: {source}", args.join(" "))]
    Spawn {
        /// Provider binary that failed to spawn.
        provider: String,
        /// Arguments passed to the provider.
        args: Vec<String>,
        /// Underlying OS error.
        source: std::io::Error,
    },

    /// The provider ran but exited non-zero.
    #[error("`{provider} {}` failed (exit {code}): {stderr}", args.join(" "))]
    Command {
        /// Provider binary.
        provider: String,
        /// Arguments passed to the provider.
        args: Vec<String>,
        /// Exit code (or -1 when terminated by a signal).
        code: i32,
        /// Captured stderr, trimmed.
        stderr: String,
    },
}

/// Result alias for container operations.
pub type Result<T> = std::result::Result<T, ContainerError>;

/// Whether `provider` resolves to podman (by binary name or path
/// basename). podman supports `--group-add keep-groups`, which docker
/// does not, so callers branch their GPU group passthrough on this.
fn provider_is_podman(provider: &str) -> bool {
    std::path::Path::new(provider)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(provider)
        .contains("podman")
}

/// Host AMD GPU device nodes to expose to a node container (`--device`)
/// when host GPU access is requested: the KFD compute device
/// (`/dev/kfd`) and the DRM render nodes (`/dev/dri`). Only paths that
/// actually exist on the host are returned, so a host missing one simply
/// omits it.
fn host_gpu_devices() -> Vec<String> {
    ["/dev/kfd", "/dev/dri"]
        .iter()
        .filter(|p| std::path::Path::new(p).exists())
        .map(|p| (*p).to_string())
        .collect()
}

/// Supplementary groups that own the host GPU device nodes
/// (`--group-add`). `video` and `render` are the conventional owners of
/// `/dev/kfd` and the `/dev/dri/render*` nodes on ROCm hosts; docker is
/// given these explicitly (podman inherits them via `keep-groups`).
fn host_gpu_groups() -> Vec<String> {
    vec!["video".to_string(), "render".to_string()]
}

/// A phase of container bring-up, reported to the `progress` callback of
/// [`Engine::bring_up`] so the host can surface detailed, live status to
/// clients as a session starts.
///
/// Each variant maps to a `(state, message)` pair via [`Self::health`],
/// keeping the full set of bring-up conditions described in one place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BringUpPhase {
    /// The image is already present locally, so the pull is skipped.
    ImagePresent { image: String },
    /// Pulling the image from its registry (can take a while).
    Pulling { image: String },
    /// The image pull finished successfully.
    Pulled { image: String },
    /// Reusing a per-session network that already exists.
    NetworkExists { network: String },
    /// Creating the per-session network.
    CreatingNetwork { network: String },
    /// Starting node container `rank` (0-based) of `total`.
    LaunchingNode { rank: u32, total: u32, name: String },
    /// Node container `rank` (0-based) of `total` has started.
    NodeStarted { rank: u32, total: u32, name: String },
}

impl BringUpPhase {
    /// The lifecycle `state` slug for this phase: one of `"pulling"`,
    /// `"networking"`, or `"starting"`. Stable enough for clients to key
    /// off while [`message`](Self::message) carries the human detail.
    pub fn state(&self) -> &'static str {
        match self {
            BringUpPhase::ImagePresent { .. }
            | BringUpPhase::Pulling { .. }
            | BringUpPhase::Pulled { .. } => "pulling",
            BringUpPhase::NetworkExists { .. } | BringUpPhase::CreatingNetwork { .. } => {
                "networking"
            }
            BringUpPhase::LaunchingNode { .. } | BringUpPhase::NodeStarted { .. } => "starting",
        }
    }

    /// A detailed, human-readable description of this phase, suitable for
    /// surfacing directly to the user as the session's status message.
    pub fn message(&self) -> String {
        match self {
            BringUpPhase::ImagePresent { image } => {
                format!("image {image} already present locally; skipping pull")
            }
            BringUpPhase::Pulling { image } => {
                format!("pulling image {image} (this can take a while)…")
            }
            BringUpPhase::Pulled { image } => format!("image {image} ready"),
            BringUpPhase::NetworkExists { network } => {
                format!("reusing existing session network {network}")
            }
            BringUpPhase::CreatingNetwork { network } => {
                format!("creating session network {network}")
            }
            BringUpPhase::LaunchingNode { rank, total, name } => {
                format!("starting node {}/{total} ({name})", rank + 1)
            }
            BringUpPhase::NodeStarted { rank, total, name } => {
                format!("node {}/{total} ({name}) started", rank + 1)
            }
        }
    }

    /// Convenience pairing of [`state`](Self::state) and
    /// [`message`](Self::message).
    pub fn health(&self) -> (&'static str, String) {
        (self.state(), self.message())
    }
}

/// A resolved container provider plus the operations mirage performs on
/// it. Cheap to clone; holds only the provider binary name/path.
#[derive(Debug, Clone)]
pub struct Engine {
    provider: String,
}

impl Engine {
    /// Resolve an engine for a containerised profile, applying the
    /// "explicit > `MIRAGE_CONTAINER_PROVIDER` > autodetect (podman then
    /// docker)" policy. Errors with [`ContainerError::NoProvider`] when
    /// nothing is available.
    pub fn resolve(def: &ContainerizedDef) -> Result<Self> {
        let provider = mirage_core::container::resolve_provider(def.provider.as_deref())
            .ok_or(ContainerError::NoProvider)?;
        Ok(Self { provider })
    }

    /// Build an engine around an explicit provider binary (name or
    /// path). Primarily for tests and callers that already resolved a
    /// provider.
    pub fn with_provider(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
        }
    }

    /// The resolved provider binary (`"podman"`, `"docker"`, or a path).
    pub fn provider(&self) -> &str {
        &self.provider
    }

    // ---- argv builders (pure) -------------------------------------

    /// Build the argv (after the provider binary) for launching a
    /// detached node container.
    ///
    /// `command` is the container's foreground process (PID 1). Mirage
    /// runs each node's own `mirage host --session <id> --rank <n>` here
    /// so the container hosts its node directly; an empty `command`
    /// leaves the image's default entrypoint in place.
    ///
    /// When `host_gpus` is set, the container is launched with the
    /// supplementary groups needed to open the passed-through GPU device
    /// nodes. The mechanism depends on `provider`: podman inherits the
    /// launching user's groups via `--group-add keep-groups`, while
    /// docker (which has no `keep-groups`) is given the named `groups`
    /// explicitly. When `host_gpus` is unset no group passthrough is
    /// emitted, which keeps plain (non-GPU) containers working on docker
    /// — `keep-groups` is a podman-only feature and docker rejects it.
    ///
    /// The container is named and given a matching hostname so peers can
    /// resolve it by name on the shared network.
    pub fn run_argv(
        provider: &str,
        name: &str,
        image: &str,
        network: Option<&str>,
        host_gpus: bool,
        mounts: &[FileMount],
        devices: &[String],
        groups: &[String],
        env: &[(String, String)],
        command: &[String],
    ) -> Vec<String> {
        let mut argv = vec![
            "run".to_string(),
            "-d".to_string(),
            "--name".to_string(),
            name.to_string(),
            "--hostname".to_string(),
            name.to_string(),
        ];
        if host_gpus {
            // Run the GPU device nodes unconfined and grant the container
            // the supplementary groups that own `/dev/kfd` and
            // `/dev/dri/*`, so the workload can open them.
            argv.push("--security-opt".to_string());
            argv.push("seccomp=unconfined".to_string());
            if provider_is_podman(provider) {
                // podman inherits the launching user's supplementary
                // groups (including `video`/`render`) rather than naming
                // them. It also rejects combining `keep-groups` with any
                // other `--group-add`, so the named groups are dropped.
                argv.push("--group-add".to_string());
                argv.push("keep-groups".to_string());
            } else {
                // docker has no `keep-groups`; add the named GPU groups
                // explicitly so the workload can open the device nodes.
                for g in groups {
                    argv.push("--group-add".to_string());
                    argv.push(g.clone());
                }
            }
        }
        if let Some(net) = network {
            argv.push("--network".to_string());
            argv.push(net.to_string());
        }
        for (k, v) in env {
            argv.push("-e".to_string());
            argv.push(format!("{k}={v}"));
        }
        for m in mounts {
            argv.push("-v".to_string());
            argv.push(m.to_volume_arg());
        }
        for d in devices {
            argv.push("--device".to_string());
            argv.push(d.clone());
        }
        argv.push(image.to_string());
        // The container's foreground process. Mirage hosts the node from
        // inside the container, so this is normally `mirage host ...`.
        argv.extend(command.iter().cloned());
        argv
    }

    /// Build the argv (after the provider binary) for executing a
    /// command inside an already-running node container.
    ///
    /// `-i` keeps stdin open so the PTY bridge behaves the same as for a
    /// non-containerised exec; environment is injected explicitly with
    /// `-e` rather than inherited from the host.
    pub fn exec_argv(
        container: &str,
        workdir: Option<&str>,
        env: &[(String, String)],
        command: &str,
        args: &[String],
    ) -> Vec<String> {
        let mut argv = vec!["exec".to_string(), "-i".to_string()];
        if let Some(wd) = workdir {
            argv.push("-w".to_string());
            argv.push(wd.to_string());
        }
        for (k, v) in env {
            argv.push("-e".to_string());
            argv.push(format!("{k}={v}"));
        }
        argv.push(container.to_string());
        argv.push(command.to_string());
        argv.extend(args.iter().cloned());
        argv
    }

    /// Full argv including the provider binary for executing a command
    /// inside a node container. Convenience for callers that build a
    /// `Command` from a single vector.
    pub fn exec_command_line(
        &self,
        container: &str,
        workdir: Option<&str>,
        env: &[(String, String)],
        command: &str,
        args: &[String],
    ) -> Vec<String> {
        let mut full = vec![self.provider.clone()];
        full.extend(Self::exec_argv(container, workdir, env, command, args));
        full
    }

    // ---- side-effecting operations --------------------------------

    /// Pull `image` so node launches don't race on an implicit pull.
    pub fn pull(&self, image: &str) -> Result<()> {
        self.checked(&["pull".to_string(), image.to_string()])
    }

    /// Whether `image` is already present locally.
    pub fn image_present(&self, image: &str) -> bool {
        self.status(&[
            "image".to_string(),
            "inspect".to_string(),
            image.to_string(),
        ])
        .unwrap_or(false)
    }

    /// Whether a network named `name` already exists.
    pub fn network_exists(&self, name: &str) -> bool {
        self.status(&[
            "network".to_string(),
            "inspect".to_string(),
            name.to_string(),
        ])
        .unwrap_or(false)
    }

    /// Create the per-session network if it does not already exist.
    pub fn ensure_network(&self, name: &str) -> Result<()> {
        if self.network_exists(name) {
            return Ok(());
        }
        self.checked(&[
            "network".to_string(),
            "create".to_string(),
            name.to_string(),
        ])
    }

    /// Launch a detached node container and return its id (the trimmed
    /// stdout of `run -d`).
    ///
    /// `host_gpus` requests host GPU access for the container; the
    /// group passthrough it implies is provider-specific (see
    /// [`Self::run_argv`]).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_node(
        &self,
        name: &str,
        image: &str,
        network: Option<&str>,
        host_gpus: bool,
        mounts: &[FileMount],
        devices: &[String],
        groups: &[String],
        env: &[(String, String)],
        command: &[String],
    ) -> Result<String> {
        let argv = Self::run_argv(
            &self.provider,
            name,
            image,
            network,
            host_gpus,
            mounts,
            devices,
            groups,
            env,
            command,
        );
        let out = self.output(&argv)?;
        Ok(String::from_utf8_lossy(&out).trim().to_string())
    }

    /// Whether a container named `name` is currently running.
    pub fn container_running(&self, name: &str) -> bool {
        match self.output(&[
            "inspect".to_string(),
            "-f".to_string(),
            "{{.State.Running}}".to_string(),
            name.to_string(),
        ]) {
            Ok(out) => String::from_utf8_lossy(&out).trim() == "true",
            Err(_) => false,
        }
    }

    /// Best-effort removal of a single container.
    pub fn rm(&self, name: &str) {
        let _ = self.status(&["rm".to_string(), "-f".to_string(), name.to_string()]);
    }

    /// Best-effort removal of a network.
    pub fn network_rm(&self, name: &str) {
        let _ = self.status(&["network".to_string(), "rm".to_string(), name.to_string()]);
    }

    /// Pull the image, create the network, and launch one container per
    /// rank, returning the [`ContainerState`] the host should persist
    /// plus the per-rank container ids (the trimmed stdout of `run -d`).
    ///
    /// `host_gpus` requests host GPU access for every node container
    /// (the provider-specific group passthrough described on
    /// [`Self::run_argv`]); the emulator decides whether its workload
    /// needs it.
    ///
    /// `node_env(rank)` yields the environment for the node of that rank
    /// (mirage injects `MIRAGE_RANK`/`MIRAGE_HEAD_ADDR`/`MIRAGE_HEAD_PORT`
    /// there). `node_command(rank)` yields the container's foreground
    /// command for that rank (mirage runs the node's own `mirage host
    /// --rank <n>`). `progress(phase)` is invoked before/after each step
    /// ([`BringUpPhase`]) so callers can surface detailed live status.
    /// On any failure the partially-created containers and network are
    /// torn down before returning the error, so a failed bring-up never
    /// leaks resources.
    #[allow(clippy::too_many_arguments)]
    pub fn bring_up<F, G, P>(
        &self,
        session: &mirage_core::session::SessionId,
        def: &ContainerizedDef,
        host_gpus: bool,
        node_count: u32,
        head_port: u16,
        mut node_env: F,
        mut node_command: G,
        mut progress: P,
    ) -> Result<(ContainerState, Vec<(u32, String)>)>
    where
        F: FnMut(u32) -> Vec<(String, String)>,
        G: FnMut(u32) -> Vec<String>,
        P: FnMut(BringUpPhase),
    {
        let network = mirage_core::container::network_name(session);

        // Pull the image unless it is already present locally; pulling a
        // large image is the slowest, most visible step, so report it.
        if self.image_present(&def.image) {
            progress(BringUpPhase::ImagePresent {
                image: def.image.clone(),
            });
        } else {
            progress(BringUpPhase::Pulling {
                image: def.image.clone(),
            });
            self.pull(&def.image)?;
            progress(BringUpPhase::Pulled {
                image: def.image.clone(),
            });
        }

        let mut state = ContainerState {
            provider: self.provider.clone(),
            image: def.image.clone(),
            network: Some(network.clone()),
            head_port,
            nodes: Vec::new(),
        };
        let mut cids: Vec<(u32, String)> = Vec::new();

        // Helper that removes anything created so far on failure.
        let rollback = |engine: &Engine, nodes: &[NodeContainer]| {
            for n in nodes {
                engine.rm(&n.name);
            }
            engine.network_rm(&network);
        };

        if self.network_exists(&network) {
            progress(BringUpPhase::NetworkExists {
                network: network.clone(),
            });
        } else {
            progress(BringUpPhase::CreatingNetwork {
                network: network.clone(),
            });
            self.ensure_network(&network)?;
        }

        // When the emulator requested host GPU access, expose the host's
        // GPU device nodes and the groups that own them on top of any
        // devices/groups the profile already configured. The group
        // passthrough mechanism itself is provider-specific and handled
        // in `run_argv`.
        let (devices, groups) = if host_gpus {
            let mut devices = def.devices.clone();
            devices.extend(host_gpu_devices());
            let mut groups = def.groups.clone();
            groups.extend(host_gpu_groups());
            (devices, groups)
        } else {
            (def.devices.clone(), def.groups.clone())
        };

        for rank in 0..node_count {
            let name = mirage_core::container::container_name(session, rank);
            progress(BringUpPhase::LaunchingNode {
                rank,
                total: node_count,
                name: name.clone(),
            });
            let env = node_env(rank);
            let command = node_command(rank);
            match self.launch_node(
                &name,
                &def.image,
                Some(&network),
                host_gpus,
                &def.mounts,
                &devices,
                &groups,
                &env,
                &command,
            ) {
                Ok(cid) => {
                    progress(BringUpPhase::NodeStarted {
                        rank,
                        total: node_count,
                        name: name.clone(),
                    });
                    cids.push((rank, cid));
                    state.nodes.push(NodeContainer { rank, name });
                }
                Err(e) => {
                    rollback(self, &state.nodes);
                    return Err(e);
                }
            }
        }

        Ok((state, cids))
    }

    // ---- private command plumbing ---------------------------------

    /// Run the provider with `args`, succeeding only on a zero exit.
    fn checked(&self, args: &[String]) -> Result<()> {
        let output = Command::new(&self.provider)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .map_err(|source| ContainerError::Spawn {
                provider: self.provider.clone(),
                args: args.to_vec(),
                source,
            })?;
        if output.status.success() {
            Ok(())
        } else {
            Err(ContainerError::Command {
                provider: self.provider.clone(),
                args: args.to_vec(),
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        }
    }

    /// Run the provider with `args` and return whether it exited zero.
    fn status(&self, args: &[String]) -> Result<bool> {
        let status = Command::new(&self.provider)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|source| ContainerError::Spawn {
                provider: self.provider.clone(),
                args: args.to_vec(),
                source,
            })?;
        Ok(status.success())
    }

    /// Run the provider with `args`, returning captured stdout on a zero
    /// exit.
    fn output(&self, args: &[String]) -> Result<Vec<u8>> {
        let output = Command::new(&self.provider)
            .args(args)
            .stdin(Stdio::null())
            .output()
            .map_err(|source| ContainerError::Spawn {
                provider: self.provider.clone(),
                args: args.to_vec(),
                source,
            })?;
        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(ContainerError::Command {
                provider: self.provider.clone(),
                args: args.to_vec(),
                code: output.status.code().unwrap_or(-1),
                stderr: String::from_utf8_lossy(&output.stderr).trim().to_string(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    fn mount(spec: &str) -> FileMount {
        FileMount::parse(spec).unwrap()
    }

    /// Mock provider: logs every invocation to `log`, exits non-zero for
    /// `network inspect` (so `ensure_network` takes the create path) and
    /// `image inspect` (so callers think the image is absent), and prints
    /// a fake id on stdout for everything else.
    fn mock_provider(dir: &Path, log: &Path) -> std::path::PathBuf {
        let provider = dir.join("mock-provider.sh");
        let script = format!(
            "#!/bin/sh\necho \"$@\" >> {log}\n\
             if [ \"$1\" = network ] && [ \"$2\" = inspect ]; then exit 1; fi\n\
             if [ \"$1\" = image ] && [ \"$2\" = inspect ]; then exit 1; fi\n\
             echo fake-cid-123\n",
            log = log.display()
        );
        std::fs::write(&provider, script).unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        provider
    }

    #[test]
    fn run_argv_includes_network_env_and_mounts() {
        let env = vec![
            ("MIRAGE_RANK".to_string(), "0".to_string()),
            ("MIRAGE_HEAD_PORT".to_string(), "5000".to_string()),
        ];
        let mounts = vec![mount("/data:/data:ro"), mount("/h:/c")];
        let devices = vec!["/dev/kfd".to_string(), "/dev/dri".to_string()];
        let groups = vec!["video".to_string(), "render".to_string()];
        let command = vec![
            "/mnt/mirage/bin/mirage".to_string(),
            "host".to_string(),
            "--session".to_string(),
            "s".to_string(),
            "--rank".to_string(),
            "0".to_string(),
        ];
        let argv = Engine::run_argv(
            "podman",
            "mirage-s-node-0",
            "img:latest",
            Some("mirage-s"),
            true,
            &mounts,
            &devices,
            &groups,
            &env,
            &command,
        );

        let joined = argv.join(" ");
        assert!(joined.starts_with("run -d --name mirage-s-node-0 --hostname mirage-s-node-0"));
        assert!(joined.contains("--security-opt seccomp=unconfined"));
        assert!(joined.contains("--group-add keep-groups"));
        assert!(joined.contains("--network mirage-s"));
        assert!(joined.contains("-e MIRAGE_RANK=0"));
        assert!(joined.contains("-e MIRAGE_HEAD_PORT=5000"));
        assert!(joined.contains("-v /data:/data:ro"));
        assert!(joined.contains("-v /h:/c"));
        assert!(joined.contains("--device /dev/kfd"));
        assert!(joined.contains("--device /dev/dri"));
        // On podman the named groups are dropped: `--group-add
        // keep-groups` cannot be combined with other `--group-add`
        // options, and already inherits them from the host.
        assert!(!joined.contains("--group-add video"));
        assert!(!joined.contains("--group-add render"));
        assert!(joined.ends_with("img:latest /mnt/mirage/bin/mirage host --session s --rank 0"));
    }

    #[test]
    fn run_argv_docker_host_gpus_adds_named_groups() {
        // docker has no `keep-groups`; the named GPU groups are added
        // explicitly so the workload can open the device nodes.
        let groups = vec!["video".to_string(), "render".to_string()];
        let argv = Engine::run_argv(
            "docker",
            "n",
            "img",
            None,
            true,
            &[],
            &[],
            &groups,
            &[],
            &[],
        );
        let joined = argv.join(" ");
        assert!(joined.contains("--security-opt seccomp=unconfined"));
        assert!(!joined.contains("keep-groups"));
        assert!(joined.contains("--group-add video"));
        assert!(joined.contains("--group-add render"));
    }

    #[test]
    fn run_argv_without_host_gpus_omits_group_passthrough() {
        // Plain (non-GPU) containers emit no group passthrough at all, so
        // docker — which rejects `keep-groups` — keeps working.
        let groups = vec!["video".to_string()];
        let argv = Engine::run_argv(
            "docker", "n", "img", None, false, &[], &[], &groups, &[], &[],
        );
        let joined = argv.join(" ");
        assert!(!joined.contains("--group-add"));
        assert!(!joined.contains("keep-groups"));
        assert!(!joined.contains("seccomp=unconfined"));
    }

    #[test]
    fn run_argv_omits_network_when_none() {
        let command = vec!["sleep".to_string(), "infinity".to_string()];
        let argv = Engine::run_argv(
            "podman", "n", "img", None, false, &[], &[], &[], &[], &command,
        );
        assert!(!argv.iter().any(|a| a == "--network"));
        assert_eq!(argv.last().map(String::as_str), Some("infinity"));
    }

    #[test]
    fn exec_argv_has_workdir_env_and_command() {
        let env = vec![("K".to_string(), "V".to_string())];
        let argv = Engine::exec_argv(
            "mirage-s-node-1",
            Some("/work"),
            &env,
            "/bin/echo",
            &["hi".to_string(), "there".to_string()],
        );
        assert_eq!(
            argv,
            vec![
                "exec",
                "-i",
                "-w",
                "/work",
                "-e",
                "K=V",
                "mirage-s-node-1",
                "/bin/echo",
                "hi",
                "there",
            ]
        );
    }

    #[test]
    fn exec_command_line_prefixes_provider() {
        let engine = Engine::with_provider("podman");
        let line = engine.exec_command_line("c", None, &[], "ls", &[]);
        assert_eq!(line, vec!["podman", "exec", "-i", "c", "ls"]);
    }

    #[test]
    fn resolve_uses_explicit_provider() {
        let def = ContainerizedDef {
            provider: Some("docker".to_string()),
            image: "img".to_string(),
            mounts: vec![],
            devices: vec![],
            groups: vec![],
        };
        let engine = Engine::resolve(&def).unwrap();
        assert_eq!(engine.provider(), "docker");
    }

    #[test]
    fn pull_invokes_provider() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        engine.pull("img:latest").unwrap();
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(recorded.contains("pull img:latest"), "{recorded:?}");
    }

    #[test]
    fn image_present_false_when_inspect_fails() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());
        assert!(!engine.image_present("img"));
    }

    #[test]
    fn ensure_network_creates_when_absent() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        engine.ensure_network("mirage-s").unwrap();
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains("network inspect mirage-s"),
            "{recorded:?}"
        );
        assert!(recorded.contains("network create mirage-s"), "{recorded:?}");
    }

    #[test]
    fn launch_node_returns_cid() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let cid = engine
            .launch_node(
                "mirage-s-node-0",
                "img",
                Some("mirage-s"),
                false,
                &[],
                &[],
                &[],
                &[],
                &["sleep".to_string(), "infinity".to_string()],
            )
            .unwrap();
        assert_eq!(cid, "fake-cid-123");
        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(
            recorded.contains("run -d --name mirage-s-node-0"),
            "{recorded:?}"
        );
    }

    #[test]
    fn bring_up_pulls_creates_network_and_launches_each_node() {
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("log");
        let provider = mock_provider(dir.path(), &log);
        let engine = Engine::with_provider(provider.to_string_lossy().to_string());

        let session = mirage_core::session::SessionId::new("s").unwrap();
        let def = ContainerizedDef {
            provider: Some(provider.to_string_lossy().to_string()),
            image: "img:latest".to_string(),
            mounts: vec![],
            devices: vec![],
            groups: vec![],
        };

        let mut phases: Vec<BringUpPhase> = Vec::new();
        let (state, cids) = engine
            .bring_up(
                &session,
                &def,
                false,
                2,
                6000,
                |rank| vec![("MIRAGE_RANK".to_string(), rank.to_string())],
                |rank| {
                    vec![
                        "mirage".to_string(),
                        "host".to_string(),
                        "--rank".to_string(),
                        rank.to_string(),
                    ]
                },
                |phase| phases.push(phase),
            )
            .unwrap();

        assert_eq!(state.image, "img:latest");
        assert_eq!(state.network.as_deref(), Some("mirage-s"));
        assert_eq!(state.head_port, 6000);
        assert_eq!(state.nodes.len(), 2);
        assert_eq!(state.nodes[0].name, "mirage-s-node-0");
        assert_eq!(state.nodes[1].name, "mirage-s-node-1");
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0], (0, "fake-cid-123".to_string()));
        assert_eq!(cids[1], (1, "fake-cid-123".to_string()));

        let recorded = std::fs::read_to_string(&log).unwrap();
        assert!(recorded.contains("pull img:latest"), "{recorded:?}");
        assert!(recorded.contains("network create mirage-s"), "{recorded:?}");
        assert!(
            recorded.contains("run -d --name mirage-s-node-0"),
            "{recorded:?}"
        );
        assert!(
            recorded.contains("run -d --name mirage-s-node-1"),
            "{recorded:?}"
        );
        assert!(recorded.contains("-e MIRAGE_RANK=0"), "{recorded:?}");
        assert!(recorded.contains("-e MIRAGE_RANK=1"), "{recorded:?}");

        // The progress callback reports each bring-up phase in order: the
        // mock image-inspect fails, so the image is pulled, the network
        // is created, and each node is launched then confirmed started.
        assert_eq!(
            phases,
            vec![
                BringUpPhase::Pulling {
                    image: "img:latest".to_string()
                },
                BringUpPhase::Pulled {
                    image: "img:latest".to_string()
                },
                BringUpPhase::CreatingNetwork {
                    network: "mirage-s".to_string()
                },
                BringUpPhase::LaunchingNode {
                    rank: 0,
                    total: 2,
                    name: "mirage-s-node-0".to_string()
                },
                BringUpPhase::NodeStarted {
                    rank: 0,
                    total: 2,
                    name: "mirage-s-node-0".to_string()
                },
                BringUpPhase::LaunchingNode {
                    rank: 1,
                    total: 2,
                    name: "mirage-s-node-1".to_string()
                },
                BringUpPhase::NodeStarted {
                    rank: 1,
                    total: 2,
                    name: "mirage-s-node-1".to_string()
                },
            ]
        );
    }
}
